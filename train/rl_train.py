"""RL training using PPO/GRPO for SWE agentic task reinforcement.

Trains the SFT checkpoint with rewards from SWE task environments.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
from pathlib import Path
from typing import List, Dict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.model_config import TrainingConfig
from model.architecture import MoEForCausalLM, MoEConfig
from rl.environments import SWEEnvironment, EnvResult


def compute_gae(rewards: List[float], values: List[float], gamma=0.99, lam=0.95):
    gae = 0
    returns = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0.0) - values[t]
        gae = delta + gamma * lam * gae
        returns.insert(0, gae + values[t])
    return returns


def train_rl_step(
    model,
    ref_model,
    batch: Dict,
    env: SWEEnvironment,
    optimizer,
    tokenizer,
    train_config: TrainingConfig,
    device: torch.device,
):
    prompts = batch["prompts"]
    env_results: List[EnvResult] = env.run_batch(prompts)

    prompt_ids = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=train_config.rl_seq_length // 2,
    ).to(device)

    responses = [r.response for r in env_results]
    response_ids = tokenizer(
        responses, return_tensors="pt", padding=True, truncation=True,
        max_length=train_config.rl_seq_length // 2,
    ).to(device)

    rewards = torch.tensor([r.reward for r in env_results], device=device)
    scores = torch.tensor([r.score for r in env_results], device=device)

    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=response_ids.input_ids,
            attention_mask=response_ids.attention_mask,
        ).logits

    logits = model(
        input_ids=response_ids.input_ids,
        attention_mask=response_ids.attention_mask,
    ).logits

    log_probs = F.log_softmax(logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    per_token_kl = log_probs - ref_log_probs
    kl_penalty = per_token_kl.mean(dim=-1).mean(dim=-1)

    advantages = rewards - rewards.mean()
    if advantages.std() > 0:
        advantages = advantages / (advantages.std() + 1e-8)

    response_log_probs = log_probs.gather(
        -1, response_ids.input_ids.unsqueeze(-1)
    ).squeeze(-1)
    mask = response_ids.attention_mask.float()
    response_log_probs = (response_log_probs * mask).sum(-1) / mask.sum(-1)

    ratio = torch.exp(response_log_probs - response_log_probs.detach())
    clipped_ratio = torch.clamp(ratio, 1 - train_config.rl_clip_epsilon, 1 + train_config.rl_clip_epsilon)
    pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    kl_loss = (kl_penalty * train_config.rl_kl_coeff).mean()
    loss = pg_loss + kl_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "reward": rewards.mean().item(),
        "score": scores.mean().item(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = True
    train_config = TrainingConfig()

    if is_main:
        wandb.init(project=train_config.wandb_project, config={"phase": "rl"})

    model = MoEForCausalLM(MoEConfig()).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    ref_model = MoEForCausalLM(MoEConfig()).to(device)

    ckpt_dir = Path(train_config.output_dir) / "sft"
    sft_ckpt = ckpt_dir / "final.pt"
    if sft_ckpt.exists():
        sd = torch.load(sft_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        ref_model.load_state_dict(sd, strict=False)
        if is_main:
            print(f"Loaded SFT checkpoint from {sft_ckpt}")
    else:
        if is_main:
            print("No SFT checkpoint found, training RL from scratch")

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.rl_lr, weight_decay=0.01
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token

    env = SWEEnvironment()

    if is_main:
        print(f"Starting RL training for {train_config.rl_steps} steps...")

    global_step = 0

    prompt_pool = [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Debug this code: def foo(x): return x / 0",
        "Refactor this function to use async/await: ...",
        "Write a bash script to find all files over 100MB in /home and compress them.",
        "Explain the time complexity of quicksort and implement it in Rust.",
        "Create a git bisect script to find the commit that introduced a bug.",
        "Review this PR and identify any security issues: ...",
        "Write a pytest test suite for a REST API endpoint.",
        "Optimize this SQL query that's taking >10s: SELECT * FROM orders JOIN users ...",
        "Implement a rate limiter in Go using channels.",
    ]

    for step in range(train_config.rl_steps):
        prompts = random.sample(prompt_pool, min(train_config.rl_batch_size, len(prompt_pool)))
        batch = {"prompts": prompts}

        metrics = train_rl_step(
            model, ref_model, batch, env, optimizer, tokenizer,
            train_config, device,
        )

        global_step += 1

        if is_main and global_step % train_config.logging_steps == 0:
            print(
                f"Step {global_step}/{train_config.rl_steps} | "
                f"loss: {metrics['loss']:.4f} | reward: {metrics['reward']:.4f} | "
                f"score: {metrics['score']:.4f} | kl: {metrics['kl_loss']:.4f}"
            )
            wandb.log(metrics)

        if is_main and global_step % train_config.save_steps == 0:
            output_dir = Path(train_config.output_dir) / "rl"
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / f"checkpoint-{global_step}.pt")

    if is_main:
        output_dir = Path(train_config.output_dir) / "rl"
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "final.pt")
        wandb.finish()
        print("RL training complete!")


if __name__ == "__main__":
    main()
