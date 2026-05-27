"""Pretraining: next-token prediction on diverse SWE/math/language corpus.

Runs via torchrun with FSDP for memory-efficient MoE training.
"""

import os
import sys
import math
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import wandb
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.model_config import MoEModelConfig, TrainingConfig
from model.architecture import MoEForCausalLM, MoEConfig
from data.pretraining_dataset import create_pretraining_dataloader, SWE_LANGUAGES


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def get_fsdp_config(model_config, train_config):
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_available else torch.float16
    return {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        ),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "forward_prefetch": True,
        "cpu_offload": None,
        "auto_wrap_policy": transformer_auto_wrap_policy,
        "transformer_layer_cls": ["TransformerBlock"],
        "limit_all_gathers": True,
        "use_orig_params": False,
    }


def train_step(model, batch, optimizer, scheduler, scaler, grad_accum):
    loss = 0
    micro_bsz = batch["input_ids"].shape[0] // grad_accum

    model.train()
    for micro_idx in range(grad_accum):
        st = micro_idx * micro_bsz
        en = st + micro_bsz
        micro_batch = {
            k: v[st:en] for k, v in batch.items()
        }

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**micro_batch)
            micro_loss = outputs.loss / grad_accum

        scaler.scale(micro_loss).backward()
        loss += micro_loss.detach()

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

    return loss


def main():
    local_rank, world_size = setup_distributed()
    is_main = local_rank == 0

    model_config = MoEModelConfig()
    train_config = TrainingConfig()

    if is_main:
        wandb.init(project=train_config.wandb_project, config={
            "phase": "pretrain",
            "model_total_params": model_config.total_params,
            "model_activated_params": model_config.activated_params,
            **{k: v for k, v in train_config.__dict__.items() if not k.startswith("_")},
        })

    torch.manual_seed(42)
    hf_config = MoEConfig(**model_config.__dict__)

    if is_main:
        print(f"Initializing {model_config.num_hidden_layers}L MoE model...")
        print(f"  Total params: {model_config.total_params:.2f}B")
        print(f"  Activated per token: {model_config.activated_params:.2f}B")
        print(f"  Experts: {model_config.num_experts}, top-{model_config.num_experts_per_tok}")

    model = MoEForCausalLM(hf_config)
    model = model.cuda(local_rank)
    model = FSDP(model, **get_fsdp_config(model_config, train_config))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.pretrain_lr,
        weight_decay=train_config.pretrain_weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler = ShardedGradScaler(enabled=(not torch.cuda.is_bf16_supported()))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = create_pretraining_dataloader(
        tokenizer=tokenizer,
        batch_size=train_config.pretrain_batch_size,
        seq_length=train_config.pretrain_seq_length,
    )

    total_steps = train_config.pretrain_steps
    warmup_steps = train_config.pretrain_warmup_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    output_dir = Path(train_config.output_dir) / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"Starting pretraining for {total_steps} steps...")
        print(f"  Batch size: {train_config.pretrain_batch_size}")
        print(f"  Grad accum: {train_config.pretrain_grad_accum}")
        print(f"  Effective batch: {train_config.pretrain_batch_size * train_config.pretrain_grad_accum * world_size}")
        print(f"  LR: {train_config.pretrain_lr}")

    global_step = 0
    optimizer.zero_grad()

    for batch in dataloader:
        batch = {k: v.cuda(local_rank) for k, v in batch.items()}
        loss = train_step(model, batch, optimizer, scheduler, scaler, train_config.pretrain_grad_accum)

        global_step += 1

        if is_main and global_step % train_config.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {global_step}/{total_steps} | loss: {loss.item():.4f} | lr: {lr:.2e}")
            wandb.log({"loss": loss.item(), "lr": lr, "step": global_step})

        if is_main and global_step % train_config.save_steps == 0:
            torch.save(model.state_dict(), output_dir / f"checkpoint-{global_step}.pt")
            print(f"Saved checkpoint at step {global_step}")

        if global_step >= total_steps:
            break

    if is_main:
        torch.save(model.state_dict(), output_dir / "final.pt")
        wandb.finish()
        print("Pretraining complete!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
