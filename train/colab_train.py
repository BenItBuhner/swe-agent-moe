"""Single-GPU training entry point for Colab (T4/V100/A100).

Handles hardware detection, CPU offloading, gradient checkpointing,
and adjusts batch size to fit available GPU memory.
"""

import os, sys, math, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.model_config import MoEModelConfig, TrainingConfig
from model.architecture import MoEForCausalLM, MoEConfig
from data.pretraining_dataset import create_pretraining_dataloader


def get_available_memory():
    if not torch.cuda.is_available():
        return 0, 0
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    free = total - torch.cuda.memory_allocated(0) / 1e9
    return total, free


def estimate_optimal_batch_size(model, seq_len, target_mem_frac=0.7):
    total_gb, free_gb = get_available_memory()
    if total_gb == 0:
        return 1, True
    
    # Rough estimate: each token requires ~2 * hidden_size bytes for activations
    hidden = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    
    # Memory per sample (activations): ~seq_len * hidden * num_layers * 2 * 2 (bf16)
    bytes_per_sample = seq_len * hidden * num_layers * 4
    
    # Available memory for activations (after params and optimizer)
    param_mem = sum(p.numel() * 2 for p in model.parameters()) / 1e9  # BF16
    opt_mem = param_mem * 2  # AdamW states (FP32)
    
    # Overhead
    overhead = 2.0  # GB for CUDA context, etc.
    
    available = free_gb - overhead
    max_batch = max(1, int(available * target_mem_frac / (bytes_per_sample / 1e9)))
    
    return max_batch, free_gb > param_mem + opt_mem + overhead


def main():
    cpu_only = "--cpu" in sys.argv
    model_config = MoEModelConfig()
    train_config = TrainingConfig()
    
    device = torch.device("cpu")
    use_cpu_offload = False
    
    if cpu_only:
        print("CPU-only mode")
    elif torch.cuda.is_available():
        total_gb, _ = get_available_memory()
        device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_gb:.1f} GB")
        
        # Check if model fits in GPU memory
        param_mem_gb = model_config.total_params * 2  # BF16
        opt_mem_gb = model_config.total_params * 4  # AdamW FP32 copies
        total_needed = param_mem_gb + opt_mem_gb + 2.0  # overhead
        
        if total_needed > total_gb:
            use_cpu_offload = True
            print(f"Model too large for GPU ({total_needed:.1f}GB needed, {total_gb:.1f}GB available)")
            print("Using CPU offloading for optimizer states")
    
    # Initialize model
    print(f"\nInitializing {model_config.num_hidden_layers}L MoE model...")
    print(f"  Total params: {model_config.total_params:.2f}B")
    print(f"  Activated per token: {model_config.activated_params:.2f}B")
    
    hf_config = MoEConfig(**model_config.__dict__)
    model = MoEForCausalLM(hf_config)
    
    if torch.cuda.is_available() and not use_cpu_offload:
        model = model.to(device)
    else:
        model = model.to(device)  # CPU or GPU
    
    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Optimizer with CPU offload if needed
    if use_cpu_offload:
        print("Enabling CPU offload for optimizer...")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, BackwardPrefetch
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        
        model = model.to("cuda:0")
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            cpu_offload=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.pretrain_lr, weight_decay=0.1)
        scaler = ShardedGradScaler()
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.pretrain_lr,
            weight_decay=train_config.pretrain_weight_decay,
            betas=(0.9, 0.95),
        )
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    batch_size = train_config.pretrain_batch_size
    seq_length = train_config.pretrain_seq_length
    
    if torch.cuda.is_available() and not use_cpu_offload:
        est_batch, fits = estimate_optimal_batch_size(model, seq_length)
        batch_size = min(batch_size, max(1, est_batch))
        if not fits:
            use_cpu_offload = True
            print("Model doesn't fit GPU. Switching to CPU offload strategy.")
    
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: {seq_length}")
    
    dataloader = create_pretraining_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_length=seq_length,
    )
    
    # Scheduler
    total_steps = min(train_config.pretrain_steps, 1000)  # Cap for Colab testing
    warmup_steps = train_config.pretrain_warmup_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Output
    output_dir = Path(train_config.output_dir) / "pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting Colab training for {total_steps} steps...")
    print(f"  Device: {device}")
    print(f"  Batch: {batch_size}, Accum: {train_config.pretrain_grad_accum}")
    print(f"  Effective batch: {batch_size * train_config.pretrain_grad_accum}")
    
    # Training loop
    global_step = 0
    model.train()
    optimizer.zero_grad()
    
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Micro-batch gradient accumulation
        loss = 0.0
        micro_bsz = batch["input_ids"].shape[0] // train_config.pretrain_grad_accum
        
        for micro_idx in range(train_config.pretrain_grad_accum):
            st = micro_idx * micro_bsz
            en = st + micro_bsz
            micro_batch = {k: v[st:en] for k, v in batch.items()}
            
            if scaler:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**micro_batch)
                    micro_loss = outputs.loss / train_config.pretrain_grad_accum
                scaler.scale(micro_loss).backward()
            else:
                outputs = model(**micro_batch)
                micro_loss = outputs.loss / train_config.pretrain_grad_accum
                micro_loss.backward()
            
            loss += micro_loss.detach()
        
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        scheduler.step()
        
        global_step += 1
        
        if global_step % train_config.logging_steps == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {global_step}/{total_steps} | loss: {loss.item():.4f} | lr: {lr:.2e}")
        
        if global_step % train_config.save_steps == 0:
            path = output_dir / f"checkpoint-{global_step}.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved checkpoint: {path}")
        
        if global_step >= total_steps:
            break
    
    # Save final
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
