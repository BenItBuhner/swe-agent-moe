"""Evaluate the trained model on SWE benchmarks.

Tests: code generation, debugging, code review, bash scripting.
"""

import sys
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture import MoEForCausalLM, MoEConfig
from configs.model_config import MoEModelConfig
from rl.environments import SWEEnvironment


def evaluate_model(
    checkpoint_path: str,
    results_path: Optional[str] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = MoEModelConfig()
    hf_config = MoEConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.num_kv_heads,
        num_experts=model_config.num_experts,
        num_experts_per_tok=model_config.num_experts_per_tok,
        max_position_embeddings=model_config.max_position_embeddings,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token

    model = MoEForCausalLM(hf_config)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    env = SWEEnvironment()
    results = []

    for i, task in enumerate(env.tasks):
        prompt = task.get_prompt()
        print(f"\n[{i+1}/{len(env.tasks)}] Evaluating task...")
        print(f"  Prompt: {prompt[:100]}...")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip()

        score = task.score_response(response)
        reward = task.get_reward(response)

        results.append({
            "task_idx": i,
            "task_type": type(task).__name__,
            "prompt": prompt,
            "response": response,
            "score": score,
            "reward": reward,
        })

        print(f"  Score: {score:.3f} | Reward: {reward:.3f}")
        print(f"  Response preview: {response[:100]}...")

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)

    summary = {
        "checkpoint": checkpoint_path,
        "num_tasks": len(results),
        "average_score": avg_score,
        "average_reward": avg_reward,
        "results": results,
    }

    print(f"\n{'='*50}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"  Tasks evaluated: {len(results)}")
    print(f"  Average score:   {avg_score:.3f}")
    print(f"  Average reward:  {avg_reward:.3f}")
    print(f"{'='*50}")

    if results_path:
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to {results_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SWE-Agent MoE model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--results", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.results, args.max_new_tokens, args.temperature)
