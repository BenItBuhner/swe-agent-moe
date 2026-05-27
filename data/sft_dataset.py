"""SFT data pipeline using DataClaw datasets from HuggingFace.

Primarily loads:
- DataClaw/AgentClaw - general agent trajectories
- DataClaw/CodeClaw - code generation/editing
- DataClaw/SWEClaw - software engineering tasks
- DataClaw/ReasonClaw - reasoning chains

Each dataset has chat-format examples with tool-use traces.
"""

from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
from typing import Dict, Optional, Iterator
import torch
import random


class SFTDataset(IterableDataset):
    """DataClaw SFT datasets mixed for SWE/agent post-training."""

    SOURCES = {
        "DataClaw/AgentClaw": 0.35,
        "DataClaw/CodeClaw": 0.25,
        "DataClaw/SWEClaw": 0.30,
        "DataClaw/ReasonClaw": 0.10,
    }

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

    def __iter__(self):
        sources, weights = [], []
        for src, wgt in self.SOURCES.items():
            try:
                ds = load_dataset(src, split="train", streaming=True)
                sources.append(ds)
                weights.append(wgt)
            except Exception:
                continue

        if not sources:
            return

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
