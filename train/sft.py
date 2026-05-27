"""SFT (Supervised Fine-Tuning) on DataClaw datasets.

Continues from the midtrained checkpoint; fine-tunes on
AgentClaw, CodeClaw, SWEClaw, and ReasonClaw mixtures.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.model_config import TrainingConfig
from model.architecture import MoEForCausalLM, MoEConfig
from data.sft_dataset import create_sft_dataloader


def load_midtrain_checkpoint(model, checkpoint_dir: str, local_rank: int):
    ckpt_path = Path(checkpoint_dir) / "final.pt"
    state_dict = torch.load(ckpt_path, map_location=f"cuda:{local_rank}", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    is_main = local_rank == 0

    train_config = TrainingConfig()

    if is_main:
        wandb.init(project=train_config.wandb_project, config={
            "phase": "sft",
            "sft_lr": train_config.sft_lr,
            "sft_steps": train_config.sft_steps,
        })

    torch.manual_seed(42)

    model = MoEForCausalLM(MoEConfig())
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

    ckpt_dir = Path(train_config.output_dir) / "midtrain"
    if ckpt_dir.exists():
        model = load_midtrain_checkpoint(model, ckpt_dir, local_rank)
        if is_main:
            print(f"Loaded midtrain checkpoint from {ckpt_dir / 'final.pt'}")
    else:
        if is_main:
            print("No midtrain checkpoint found, starting SFT from scratch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.sft_lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    scaler = ShardedGradScaler(enabled=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' }}"
        "{% endfor %}"
    )

    dataloader = create_sft_dataloader(
        tokenizer=tokenizer,
        batch_size=train_config.sft_batch_size,
        seq_length=train_config.sft_seq_length,
    )

    total_steps = train_config.sft_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, train_config.sft_warmup_steps, total_steps
    )

    output_dir = Path(train_config.output_dir) / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Starting SFT for {total_steps} steps...")
        print(f"  LR: {train_config.sft_lr}")
        print(f"  Seq length: {train_config.sft_seq_length}")
        print(f"  Sources: DataClaw/AgentClaw, DataClaw/CodeClaw, DataClaw/SWEClaw, DataClaw/ReasonClaw")

    global_step = 0
    model.train()

    for batch in dataloader:
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
            wandb.log({"sft_loss": loss.item(), "lr": lr, "step": global_step})

        if is_main and global_step % train_config.save_steps == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint-{global_step}.pt")

        if global_step >= total_steps:
            break

    if is_main:
        torch.save(model.state_dict(), output_dir / "final.pt")
        wandb.finish()
        print("SFT complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
