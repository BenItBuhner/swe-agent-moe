"""Midtraining: continued pretraining on reasoning data (chain-of-thought, math, code reasoning).

Loads the pretrained checkpoint and trains on reasoning-focused data
to bridge from pretraining to SFT. Supports checkpoint resume.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.model_config import MoEModelConfig, TrainingConfig
from model.architecture import MoEForCausalLM, MoEConfig
from data.midtraining_dataset import create_midtraining_dataloader


def load_pretrained_checkpoint(model, checkpoint_dir, local_rank):
    ckpt_path = Path(checkpoint_dir) / "final.pt"
    state_dict = torch.load(ckpt_path, map_location=f"cuda:{local_rank}", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


def save_checkpoint(model, optimizer, scheduler, scaler, global_step, output_dir, is_main):
    if not is_main:
        return
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
    }, ckpt_dir / "training_state.pt")
    with open(output_dir / "latest_checkpoint.txt", "w") as f:
        f.write(str(global_step))
    print(f"Saved midtrain checkpoint at step {global_step}")


def load_resume_checkpoint(output_dir, model, optimizer, scheduler, scaler, local_rank, is_main):
    latest_file = output_dir / "latest_checkpoint.txt"
    if not latest_file.exists():
        return 0
    try:
        step = int(latest_file.read_text().strip())
    except (ValueError, OSError):
        return 0
    ckpt_dir = output_dir / f"checkpoint-{step}"
    if not (ckpt_dir / "model.pt").exists():
        return 0
    state_dict = torch.load(ckpt_dir / "model.pt", map_location=f"cuda:{local_rank}", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    ts = torch.load(ckpt_dir / "training_state.pt", map_location=f"cuda:{local_rank}", weights_only=False)
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    scaler.load_state_dict(ts["scaler"])
    if is_main:
        print(f"Resumed midtrain from checkpoint at step {step}")
    return step


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    is_main = local_rank == 0

    model_config = MoEModelConfig()
    train_config = TrainingConfig()

    if is_main:
        wandb.init(project=train_config.wandb_project, config={"phase": "midtrain"})

    torch.manual_seed(42)

    hf_config = model_config.to_hf_config()
    model = MoEForCausalLM(hf_config)
    model = model.cuda(local_rank)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        auto_wrap_policy=transformer_auto_wrap_policy,
        limit_all_gathers=True,
    )

    ckpt_dir = Path(train_config.output_dir) / "pretrain"
    if ckpt_dir.exists():
        model = load_pretrained_checkpoint(model, ckpt_dir, local_rank)
        if is_main:
            print(f"Loaded pretrained checkpoint from {ckpt_dir / 'final.pt'}")
    else:
        if is_main:
            print("No pretrained checkpoint found, starting midtrain from scratch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.midtrain_lr, weight_decay=0.1)
    scaler = ShardedGradScaler(enabled=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token

    total_steps = train_config.midtrain_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_config.midtrain_warmup_steps, total_steps
    )

    output_dir = Path(train_config.output_dir) / "midtrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    global_step = load_resume_checkpoint(output_dir, model, optimizer, scheduler, scaler, local_rank, is_main)
    if global_step > 0:
        for _ in range(global_step):
            scheduler.step()
    optimizer.zero_grad()

    dataloader = create_midtraining_dataloader(
        tokenizer=tokenizer,
        batch_size=train_config.midtrain_batch_size,
        seq_length=train_config.midtrain_seq_length,
    )

    if is_main:
        print(f"Starting midtraining from step {global_step} to {total_steps}...")
        print(f"  LR: {train_config.midtrain_lr}")
        print(f"  Seq length: {train_config.midtrain_seq_length}")
        print(f"  Sources: {list(train_config.midtrain_data_mix)}")

    model.train()

    for batch in dataloader:
        if global_step >= total_steps:
            break

        batch = {k: v.cuda(local_rank) for k, v in batch.items()}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        global_step += 1

        if is_main and global_step % train_config.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {global_step}/{total_steps} | loss: {loss.item():.4f} | lr: {lr:.2e}")
            wandb.log({"midtrain_loss": loss.item(), "lr": lr, "step": global_step})

        if global_step % train_config.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, global_step, output_dir, is_main)

    if is_main:
        torch.save(model.state_dict(), output_dir / "final.pt")
        wandb.finish()
        print("Midtraining complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
