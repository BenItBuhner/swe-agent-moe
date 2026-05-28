"""Build curated DataClaw SFT datasets from raw DataClaw exports on HuggingFace.

Constructs four datasets from raw agent-coding conversation logs:
  - AgentClaw: agent orchestration / tool coordination
  - CodeClaw: code generation and editing
  - SWEClaw: software engineering tasks (debug, test, review, CI)
  - ReasonClaw: reasoning chains, planning, analysis

Raw sources (multiple individuals' DataClaw exports):
  - woctordho/dataclaw, peteromallet/dataclaw-peteromallet, etc.

Usage:
    python data/build_dataclaw.py                         # build + save locally
    python data/build_dataclaw.py --push                  # push to HF hub
    python data/build_dataclaw.py --push --namespace my-org  # push to custom ns
"""

import json, re, os, sys, math, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Iterator, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Raw DataClaw exports on HF (sorted by quality / size) ──────────────
RAW_SOURCES = [
    "woctordho/dataclaw",               # 486 sessions, multi-model
    "woctordho/dataclaw-windows",        # more sessions, multi-model
    "peteromallet/dataclaw-peteromallet",  # 549 sessions, mostly Claude
    "peteromallet/my-dataclaw-data",      # 1K+ sessions, multi-model
    "zhiyaowang/dataclaw-zhiyaowang",     # multi-model, diverse projects
    "Codingxx/dataclaw-peteromallet",     # 549 sessions
    "nathanstvnsn/dataclaw-peteromallet", # 549 sessions
    "A99311/my-dataclaw-data",           # additional sessions
]

# ── Keywords for classifying conversations ──────────────────────────────
AGENT_KW = [
    "delegate", "subagent", "coordinator", "orchestrat", "workflow",
    "pipeline", "multi-step", "parallel", "tool use", "function call",
    "autonomous", "agent", "plan then", "break down", "sub-task",
]

CODE_KW = [
    "implement", "write a function", "create class", "generate code",
    "code review", "refactor", "add feature", "build a", "develop",
    "coding", "program", "script", "function that", "class that",
]

SWE_KW = [
    "debug", "bug", "fix", "error", "issue", "test", "unit test",
    "integration test", "CI", "pipeline", "deploy", "commit", "PR",
    "pull request", "merge", "git", "branch", "version control",
    "code review", "lint", "type check", "compilation", "build error",
]

REASON_KW = [
    "think", "reason", "explain", "why", "how does", "analyze",
    "compare", "contrast", "evaluate", "prove", "deduce", "infer",
    "solve", "calculate", "compute", "derive", "proof",
]

# ── Classification logic ────────────────────────────────────────────────


def classify_conversation(messages: List[Dict]) -> str:
    text = ""
    for m in messages:
        content = m.get("content", "") or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        text += content.lower() + " "

    scores = {k: 0 for k in ["agent", "code", "swe", "reason"]}
    scores["agent"] = sum(1 for kw in AGENT_KW if kw in text)
    scores["code"] = sum(1 for kw in CODE_KW if kw in text)
    scores["swe"] = sum(1 for kw in SWE_KW if kw in text)
    scores["reason"] = sum(1 for kw in REASON_KW if kw in text)

    # Boost 'code' for code-heavy conversations
    code_indicators = ["```", "def ", "class ", "import ", "return ",
                       "function ", "const ", "let ", "var ", "#include"]
    for ind in code_indicators:
        if ind in text:
            scores["code"] += 2

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "code"  # default: generic code
    return best


# ── Message formatting ──────────────────────────────────────────────────


def flatten_content_parts(m: Dict) -> str:
    content = m.get("content", "") or ""
    parts = m.get("content_parts") or []
    if not parts:
        return str(content)
    texts = [str(content)] if content else []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text":
            texts.append(p.get("text", ""))
        elif isinstance(p, dict) and p.get("type") == "image":
            texts.append("[image]")
    return "\n".join(texts)


def format_messages(messages: List[Dict]) -> List[Dict]:
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = flatten_content_parts(m)
        tool_uses = m.get("tool_uses") or m.get("tool_calls") or []

        if tool_uses and role == "assistant":
            tool_text = []
            for tu in tool_uses:
                tool_name = tu.get("tool", tu.get("name", "tool"))
                inp = tu.get("input", {})
                output = tu.get("output", {})
                if isinstance(output, dict):
                    out_text = output.get("text", "") or str(output)[:200]
                else:
                    out_text = str(output)[:200]
                # Include tool call + result as structured text
                tool_text.append(
                    f"<tool>{tool_name}</tool>\n"
                    f"<input>{json.dumps(inp, ensure_ascii=False)[:500]}</input>\n"
                    f"<output>{out_text[:500]}</output>"
                )
            if tool_text:
                content += "\n\n" + "\n\n".join(tool_text)

        if not content:
            continue
        result.append({"role": role, "content": content})
    return result


# ── Dataset builder ─────────────────────────────────────────────────────


def load_raw_sources() -> Iterator[Tuple[str, str, List[Dict]]]:
    """Yield (source_name, classification, formatted_messages) tuples."""
    from datasets import load_dataset

    loaded = 0
    for source in RAW_SOURCES:
        try:
            ds = load_dataset(source, split="train", streaming=True)
            print(f"  Loading {source}...")
        except Exception as e:
            print(f"  Skipping {source}: {e}")
            continue

        count = 0
        for example in ds:
            messages = example.get("messages") or example.get("conversations") or []
            if not messages or len(messages) < 2:
                continue

            classification = classify_conversation(messages)
            formatted = format_messages(messages)
            if not formatted or len(formatted) < 2:
                continue

            # Estimate quality: prefer longer conversations with tool use
            tool_count = sum(1 for m in messages if m.get("tool_uses") or m.get("tool_calls"))
            quality = min(len(messages) / 5 + tool_count, 10)

            yield (source, classification, formatted)
            count += 1
            loaded += 1

            if count >= 200:  # limit per source for balance
                break

        print(f"    → {count} conversations loaded")

    print(f"\n  Total loaded: {loaded} conversations")


def deduplicate(convs: List[Tuple[str, List[Dict]]]) -> List[Tuple[str, List[Dict]]]:
    seen = set()
    result = []
    for label, msgs in convs:
        # Hash based on first user message content
        first_user = ""
        for m in msgs:
            if m.get("role") == "user":
                first_user = m.get("content", "")[:200]
                break
        key = hashlib.md5(first_user.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append((label, msgs))
    return result


def build_datasets(
    output_dir: str = "data/dataclaw_built",
    max_per_class: int = 2000,
    dedup: bool = True,
):
    """Build the four DataClaw datasets and save to disk."""
    from datasets import Dataset, Features, Value, Sequence

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading raw DataClaw exports from HuggingFace...")
    all_convs = list(load_raw_sources())
    print(f"\nLoaded {len(all_convs)} total conversations")

    # Separate by classification
    classified = defaultdict(list)
    for source, classification, messages in all_convs:
        classified[classification].append((source, messages))

    # Build each dataset
    dataset_map = {
        "agent": ("AgentClaw", "Agent orchestration and tool coordination conversations"),
        "code": ("CodeClaw", "Code generation and editing conversations"),
        "swe": ("SWEClaw", "Software engineering task conversations"),
        "reason": ("ReasonClaw", "Reasoning and analysis conversations"),
    }

    stats = {}
    for key, (ds_name, description) in dataset_map.items():
        convs = classified.get(key, [])
        if dedup:
            convs = deduplicate(convs)

        # Shuffle and limit
        random.shuffle(convs)
        convs = convs[:max_per_class]

        # Convert to HF Dataset
        records = []
        for source, msgs in convs:
            # Format as single text with chat template
            parts = []
            for m in msgs:
                role = m["role"]
                content = m["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(parts)

            records.append({
                "text": text,
                "messages": json.dumps(msgs, ensure_ascii=False),
                "source": source,
                "num_messages": len(msgs),
            })

        ds = Dataset.from_list(records)
        ds_path = out_path / ds_name
        ds.save_to_disk(str(ds_path))
        stats[ds_name] = len(ds)

        print(f"\n{ds_name}: {len(ds)} conversations")
        print(f"  Saved to: {ds_path}")

    # Save metadata
    meta = {
        "description": "Curated DataClaw SFT datasets for SWE/agentic model training",
        "sources": RAW_SOURCES,
        "stats": stats,
        "total_conversations": sum(stats.values()),
        "format": "chat template: <|im_start|>role\\ncontent<|im_end|>",
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"BUILD COMPLETE")
    print(f"  Total conversations: {sum(stats.values())}")
    print(f"  Output: {out_path}")
    print(f"{'='*50}")

    return stats


def push_to_hub(
    build_dir: str = "data/dataclaw_built",
    namespace: str = "barnacle-agent",
    token: Optional[str] = None,
):
    """Push built datasets to HuggingFace Hub."""
    from datasets import Dataset, load_from_disk
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    build_path = Path(build_dir)

    dataset_names = ["AgentClaw", "CodeClaw", "SWEClaw", "ReasonClaw"]

    for ds_name in dataset_names:
        ds_path = build_path / ds_name
        if not ds_path.exists():
            print(f"  Skipping {ds_name}: not found at {ds_path}")
            continue

        repo_id = f"{namespace}/{ds_name}"
        print(f"  Pushing {ds_name} → {repo_id}...")

        ds = load_from_disk(str(ds_path))
        ds.push_to_hub(repo_id, token=token, private=False)
        print(f"    Done: {len(ds)} rows")

    print("\nAll datasets pushed to HuggingFace Hub!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DataClaw SFT datasets")
    parser.add_argument("--output", default="data/dataclaw_built",
                        help="Output directory for built datasets")
    parser.add_argument("--max-per-class", type=int, default=2000,
                        help="Max conversations per class")
    parser.add_argument("--push", action="store_true",
                        help="Push datasets to HuggingFace Hub")
    parser.add_argument("--namespace", default="barnacle-agent",
                        help="HF Hub namespace for push")
    parser.add_argument("--token", default=None,
                        help="HF Hub token (or HF_TOKEN env)")

    args = parser.parse_args()

    stats = build_datasets(args.output, args.max_per_class)

    if args.push:
        token = args.token or os.environ.get("HF_TOKEN")
        push_to_hub(args.output, args.namespace, token)
