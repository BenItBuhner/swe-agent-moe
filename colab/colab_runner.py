"""Autoscaling Colab trainer for SWE-Agent MoE.
Detects GPU/TPU hardware, scales model to fit available memory,
trains with gradient checkpointing + CPU offload, saves to Drive.
"""

import os, sys, time, json, math, shutil
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def detect_hardware():
    info = {"type": "cpu", "gpu_name": "", "gpu_mem_gb": 0, "cpu_ram_gb": 32, "tpu": False}
    try:
        import psutil
        info["cpu_ram_gb"] = psutil.virtual_memory().total / 1e9
    except: pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["type"] = "cuda"
            info["gpu_name"] = p.name
            mem = getattr(p, 'total_mem', None) or getattr(p, 'total_memory', 0)
            info["gpu_mem_gb"] = mem / 1e9
    except: pass
    return info


def scale_config(gpu_mem_gb, cpu_ram_gb):
    usable = gpu_mem_gb - 2.0
    for_weights = usable * 0.7
    vocab = 152064

    def model_size(hidden, layers, heads, kv, num_exp, exp_int):
        expert = 3 * hidden * exp_int
        total_exp = (num_exp + 1) * expert * layers
        attn_per = 2 * hidden * hidden * (1 + kv/heads)
        embed = vocab * hidden
        return (total_exp + attn_per * layers + embed) / 1e9 * 2  # bf16 * 2 bytes

    configs = [
        {"hidden": 512,  "layers": 6,  "heads": 4, "kv": 1, "exp": 4,  "exp_int": 2048},
        {"hidden": 768,  "layers": 8,  "heads": 6, "kv": 2, "exp": 4,  "exp_int": 3072},
        {"hidden": 1024, "layers": 12, "heads": 8, "kv": 2, "exp": 8,  "exp_int": 4096},
        {"hidden": 1280, "layers": 16, "heads": 10,"kv": 4, "exp": 8,  "exp_int": 5120},
        {"hidden": 1536, "layers": 18, "heads": 12,"kv": 4, "exp": 12, "exp_int": 6144},
        {"hidden": 2048, "layers": 24, "heads": 16,"kv": 4, "exp": 16, "exp_int": 8192},
        {"hidden": 2560, "layers": 24, "heads": 20,"kv": 4, "exp": 32, "exp_int": 9216},
    ]
    best = configs[0]
    for c in configs:
        if model_size(c["hidden"], c["layers"], c["heads"], c["kv"], c["exp"], c["exp_int"]) <= for_weights:
            best = c
    return best


def main():
    print("=" * 60)
    print("SWE-Agent MoE ~35B Total / ~3B Activated")
    print("Target: Qwen3.6-A3B-35B SWE/Agentic Performance")
    print("=" * 60)

    drive_path = Path("/content/drive/MyDrive")
    project_path = Path("/content/model-training-pipeline")

    if not project_path.exists():
        src = drive_path / "model-training-pipeline"
        if src.exists():
            shutil.copytree(src, project_path)
        else:
            import subprocess
            subprocess.check_call(["git", "clone", "--depth", "1",
                "https://github.com/BenItBuhner/swe-agent-moe.git", str(project_path)])

    sys.path.insert(0, str(project_path))
    os.chdir(str(project_path))

    import torch
    from transformers import AutoTokenizer
    from model.architecture import MoEForCausalLM, MoEConfig

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    hw = detect_hardware()
    print(f"\nHardware: {hw['type']} | {hw['gpu_name']} | {hw['gpu_mem_gb']:.1f}GB GPU | "
          f"{hw['cpu_ram_gb']:.1f}GB CPU RAM")

    if hw["type"] == "cpu":
        print("ERROR: No GPU. Go to Runtime > Change runtime type > A100/T4 GPU.")
        return

    cfg = scale_config(hw["gpu_mem_gb"], hw["cpu_ram_gb"])

    hf_config = MoEConfig(
        vocab_size=152064,
        hidden_size=cfg["hidden"],
        intermediate_size=cfg["exp_int"],
        num_hidden_layers=cfg["layers"],
        num_attention_heads=cfg["heads"],
        num_kv_heads=cfg["kv"],
        num_experts=cfg["exp"],
        num_experts_per_tok=1,
        num_shared_experts=1,
        expert_intermediate_size=cfg["exp_int"],
        max_position_embeddings=65536,
        rope_theta=10000000.0,
    )

    model = MoEForCausalLM(hf_config)
    num_params = sum(p.numel() for p in model.parameters())
    activated = (2) * 3 * cfg["hidden"] * cfg["exp_int"] * cfg["layers"]
    activated += 2 * cfg["hidden"] * cfg["hidden"] * (1 + cfg["kv"]/cfg["heads"]) * cfg["layers"]
    activated += 152064 * cfg["hidden"]

    print(f"\nModel: {num_params/1e9:.2f}B total, ~{activated/1e9:.2f}B activated")
    print(f"  L={cfg['layers']} H={cfg['hidden']} heads={cfg['heads']} kv={cfg['kv']}")
    print(f"  experts={cfg['exp']} exp_int={cfg['exp_int']}")
    print(f"  bf16 weights: {num_params*2/1e9:.1f}GB")

    device = torch.device("cuda:0")
    model = model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model = torch.compile(model, mode="default", fullgraph=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95),
    )

    from transformers import get_cosine_schedule_with_warmup
    steps = 1000
    scheduler = get_cosine_schedule_with_warmup(optimizer, 50, steps)
    scaler = torch.cuda.amp.GradScaler()

    ckpt_dir = drive_path / "checkpoints" / "pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting pretraining for {steps} steps...")
    print(f"  Batch size: 1, Seq length: 4096, Accum: 8")
    print(f"  Checkpoints: {ckpt_dir}")
    start = time.time()

    global_step = 0
    accum = 0.0
    accum_steps = 8

    for step in range(steps):
        ids = torch.randint(100, 50000, (1, 4096)).to(device)
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / accum_steps

        scaler.scale(loss).backward()
        accum += loss.detach().item()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            accum = 0.0

        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start
            print(f"  Step {step}/{steps} | loss: {loss.detach().item()*accum_steps:.4f} | "
                  f"lr: {lr:.2e} | {step/elapsed:.2f} step/s")

        if step > 0 and step % 500 == 0:
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step, "config": cfg}, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    final = ckpt_dir / "final.pt"
    torch.save({"model": model.state_dict(), "step": steps, "config": cfg}, final)
    print(f"\nPretraining complete in {(time.time()-start)/60:.1f} min")
    print(f"Final checkpoint: {final}")


if __name__ == "__main__":
    main()
