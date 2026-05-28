"""SFT data pipeline using DataClaw datasets from HuggingFace.

Primarily loads:
- barnacle-agent/AgentClaw - general agent trajectories
- barnacle-agent/CodeClaw - code generation/editing
- barnacle-agent/SWEClaw - software engineering tasks
- barnacle-agent/ReasonClaw - reasoning chains

Each dataset has chat-format examples with tool-use traces.

Fallback sources (used when curated DataClaw datasets are unavailable):
  AgentClaw fallback → microsoft/orca-agentinstruct-1M-v1, raw DataClaw exports
  CodeClaw fallback  → codegen datasets
  SWEClaw fallback   → raw DataClaw exports
  ReasonClaw fallback → Open-Orca/OpenOrca
"""

from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from typing import Dict, Optional, Iterator, List, Tuple
import torch
import random


# ── Curated DataClaw sources (primary) ─────────────────────────────────
PRIMARY_SOURCES = {
    "barnacle-agent/AgentClaw": 0.35,
    "barnacle-agent/CodeClaw": 0.25,
    "barnacle-agent/SWEClaw": 0.30,
    "barnacle-agent/ReasonClaw": 0.10,
}

# ── Fallback sources when curated datasets aren't published yet ─────────
# These are well-known, publicly accessible datasets that fill the same roles.
FALLBACK_SOURCES: Dict[str, List[Tuple[str, float]]] = {
    "agent": [
        ("microsoft/orca-agentinstruct-1M-v1", 0.50),
        ("woctordho/dataclaw", 0.30),
        ("peteromallet/my-dataclaw-data", 0.20),
    ],
    "code": [
        ("bigcode/the-stack-v2", 0.40),
        ("codeparrot/github-code", 0.35),
        ("camel-ai/code", 0.25),
    ],
    "swe": [
        ("woctordho/dataclaw", 0.40),
        ("peteromallet/my-dataclaw-data", 0.35),
        ("woctordho/dataclaw-windows", 0.25),
    ],
    "reason": [
        ("Open-Orca/OpenOrca", 0.50),
        ("microsoft/orca-math", 0.25),
        ("jeffdshen/reher-v0.1", 0.25),
    ],
}


class SFTDataset(IterableDataset):
    """SFT data pipeline for SWE/agent post-training.

    Tries curated DataClaw datasets first; falls back to public alternatives.
    """

    SOURCES = PRIMARY_SOURCES

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_length: int = 8192,
        subset: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.subset = subset

    def _format_messages(self, example: Dict) -> Optional[str]:
        if "messages" in example:
            msgs = example["messages"]
        elif "conversations" in example:
            msgs = example["conversations"]
        elif "chat" in example:
            msgs = example["chat"]
        else:
            return None

        parts = []
        for m in msgs:
            role = m.get("role", m.get("from", "user"))
            content = m.get("content", m.get("value", ""))
            if not content:
                continue
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts) if parts else None

    def _process(self, example: Dict) -> Iterator[Dict]:
        text = self._format_messages(example)
        if text is None or len(text) < 20:
            return

        tokens = self.tokenizer(
            text, truncation=True, max_length=self.seq_length + 1,
            return_tensors="pt", padding=False,
        )["input_ids"][0]

        if len(tokens) < 10:
            return

        for i in range(0, len(tokens) - 1, self.seq_length):
            chunk = tokens[i : i + self.seq_length + 1]
            if len(chunk) < self.seq_length + 1:
                pad_len = self.seq_length + 1 - len(chunk)
                chunk = torch.cat([chunk, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}

    @staticmethod
    def _try_load(source: str) -> Optional:
        try:
            ds = load_dataset(source, split="train", streaming=True, trust_remote_code=True)
            # Quick validation: try to get one element
            for _ in ds:
                return ds
        except Exception:
            return None

    def _load_primary(self) -> Tuple[List, List]:
        sources, weights = [], []
        for src, wgt in self.SOURCES.items():
            ds = self._try_load(src)
            if ds is not None:
                sources.append(ds)
                weights.append(wgt)
                print(f"  SFT: loaded primary source {src}")
        return sources, weights

    def _load_fallback(self) -> Tuple[List, List]:
        """Try fallback sources grouped by role."""
        sources, weights = [], []
        for role, fallbacks in FALLBACK_SOURCES.items():
            loaded = False
            for src, wgt in fallbacks:
                ds = self._try_load(src)
                if ds is not None:
                    sources.append(ds)
                    weights.append(wgt * self.SOURCES.get(
                        {"agent": "barnacle-agent/AgentClaw",
                         "code": "barnacle-agent/CodeClaw",
                         "swe": "barnacle-agent/SWEClaw",
                         "reason": "barnacle-agent/ReasonClaw"}.get(role, ""), 0.25))
                    print(f"  SFT: fallback source {src} for {role}")
                    loaded = True
                    break
            if not loaded:
                print(f"  SFT: no fallback found for {role}")
        return sources, weights

    def __iter__(self):
        sources, weights = self._load_primary()
        if not sources:
            print("  SFT: primary DataClaw sources unavailable, trying fallbacks...")
            sources, weights = self._load_fallback()

        if not sources:
            print("  SFT WARNING: no data sources available!")
            return

        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        interleaved = interleave_datasets(sources, probabilities=weights, seed=42)

        for example in interleaved:
            if self.subset < 1.0 and random.random() > self.subset:
                continue
            yield from self._process(example)


def create_sft_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    seq_length: int = 8192,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SFTDataset(tokenizer, seq_length)
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_sft_collate(tokenizer.pad_token_id), pin_memory=True,
    )


def _sft_collate(pad_token_id: int):
    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = (input_ids != pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    return collate
