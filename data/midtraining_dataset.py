"""Midtraining: reasoning-focused data for chain-of-thought and SWE reasoning.

Sources:
- Open-Orca/OpenOrca (general reasoning)
- microsoft/orca-math (math reasoning)
- camel-ai/code (code understanding)
- jeffdshen/reher-v0.1 (reasoning traces)
"""

from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
from typing import Dict, Optional, Iterator
import torch
import random


class MidtrainingDataset(IterableDataset):
    """Reasoning-focused midtraining corpus."""

    SOURCES = {
        "Open-Orca/OpenOrca": 0.35,
        "microsoft/orca-math": 0.10,
        "camel-ai/code": 0.25,
        "jeffdshen/reher-v0.1": 0.30,
    }

    CHAT_TEMPLATE = (
        "<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n{answer}<|im_end|>"
    )

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_length: int = 8192,
        subset: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.subset = subset

    def _format_example(self, example: Dict) -> Optional[str]:
        if "question" in example and "answer" in example:
            return self.CHAT_TEMPLATE.format(
                question=example["question"],
                answer=example.get("response", example.get("answer", "")),
            )
        if "instruction" in example and "response" in example:
            return self.CHAT_TEMPLATE.format(
                question=example["instruction"],
                answer=example["response"],
            )
        if "messages" in example:
            msgs = example["messages"]
            formatted = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                formatted.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(formatted)
        text = example.get("text", example.get("content", ""))
        if len(text) > 50:
            return text
        return None

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, truncation=True, max_length=self.seq_length + 1,
            return_tensors="pt", padding=False,
        )["input_ids"][0]

    def _process(self, example: Dict) -> Iterator[Dict]:
        text = self._format_example(example)
        if text is None:
            return
        token_ids = self._tokenize(text)
        if len(token_ids) < 10:
            return

        for i in range(0, len(token_ids) - 1, self.seq_length):
            chunk = token_ids[i : i + self.seq_length + 1]
            if len(chunk) < self.seq_length + 1:
                pad_len = self.seq_length + 1 - len(chunk)
                chunk = torch.cat([chunk, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            yield {"input_ids": chunk[:-1], "labels": chunk[1:]}

    def __iter__(self):
        sources = []
        weights = []
        for src, wgt in self.SOURCES.items():
            try:
                ds = load_dataset(src, split="train", streaming=True)
                sources.append(ds)
                weights.append(wgt)
            except Exception as e:
                import warnings
                warnings.warn(f"Could not load {src}: {e}")
                continue

        interleaved = interleave_datasets(sources, probabilities=weights, seed=42)

        for example in interleaved:
            if self.subset < 1.0 and random.random() > self.subset:
                continue
            yield from self._process(example)


def create_midtraining_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    seq_length: int = 8192,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MidtrainingDataset(tokenizer, seq_length)
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_mid_collate(tokenizer.pad_token_id), pin_memory=True,
    )


def _mid_collate(pad_token_id: int):
    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = (input_ids != pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    return collate
