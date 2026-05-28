"""Pretraining data pipeline: SWE code, math, and general language corpus.

Loads and mixes:
- codeparrot/github-code (Python, JavaScript, TypeScript, Rust, Go, C++)
- bigcode/the-stack-v2 (broader code)
- allenai/dolma (general text)
- cerebras/SlimPajama-627B (general)
- HuggingFaceFW/fineweb-edu (educational web)
"""

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from typing import Optional, Dict, List, Iterator
import random
import math
import warnings


SWE_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Rust", "Go",
    "C++", "Java", "C", "Shell", "Ruby", "C#",
]

# Each source with its weight in the pretraining mix.
# Weights are relative; they get normalized to probabilities inside __iter__.
PRETRAIN_SOURCES = {
    "codeparrot/github-code":       0.20,
    "bigcode/the-stack-v2":         0.20,
    "allenai/dolma":                0.20,
    "cerebras/SlimPajama-627B":     0.15,
    "HuggingFaceFW/fineweb-edu":    0.25,
}

# Some datasets expose the text column under different names.
_TEXT_COLUMNS = ("text", "content", "code", "document")


class PretrainingDataset(IterableDataset):
    """Multi-source pretraining corpus with weighted mixing."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_length: int = 4096,
        subset: float = 1.0,
        language_upsample: float = 2.0,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.subset = subset
        self.language_upsample = language_upsample
        self.source_weights = source_weights or PRETRAIN_SOURCES
        self.sources = list(self.source_weights.keys())
        self.weights = list(self.source_weights.values())

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,
            return_tensors="pt",
            padding=False,
        )
        return tokens["input_ids"][0]

    def _extract_text(self, example: Dict) -> str:
        for col in _TEXT_COLUMNS:
            val = example.get(col)
            if val and isinstance(val, str):
                return val
        return ""

    def _process_example(self, example: Dict) -> Iterator[Dict]:
        text = self._extract_text(example)
        if not text or len(text) < 50:
            return

        lang = example.get("lang") or example.get("language") or ""
        is_swe = lang in SWE_LANGUAGES or any(
            kw in text[:500] for kw in ("def ", "class ", "import ", "fn ", "func ", "pub ")
        )

        token_ids = self._tokenize(text)
        if len(token_ids) < 10:
            return

        for i in range(0, len(token_ids) - 1, self.seq_length):
            chunk = token_ids[i : i + self.seq_length + 1]
            if len(chunk) < self.seq_length + 1:
                pad_len = self.seq_length + 1 - len(chunk)
                chunk = torch.cat([chunk, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            yield {
                "input_ids": chunk[:-1],
                "labels": chunk[1:],
                "is_swe": is_swe,
            }

    def __iter__(self):
        datasets_list = []
        effective_weights = []

        for source, weight in self.source_weights.items():
            try:
                ds = load_dataset(source, split="train", streaming=True, trust_remote_code=True)
                datasets_list.append(ds)
                effective_weights.append(weight)
            except Exception as e:
                warnings.warn(f"Could not load {source}: {e}", stacklevel=2)
                continue

        if not datasets_list:
            raise RuntimeError("No datasets could be loaded for pretraining")

        # Normalize weights so they sum to 1.0 (required by interleave_datasets)
        total_w = sum(effective_weights)
        effective_weights = [w / total_w for w in effective_weights]

        interleaved = interleave_datasets(
            datasets_list, probabilities=effective_weights, seed=42
        )

        for example in interleaved:
            if self.subset < 1.0 and random.random() > self.subset:
                continue
            yield from self._process_example(example)


class SWEUpweightMixin:
    """Wraps a dataset to double sample SWE content."""

    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset

    def __iter__(self):
        for batch in self.dataset:
            yield batch
            if batch.get("is_swe", False):
                yield batch  # sample twice


def create_pretraining_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    seq_length: int = 4096,
    num_workers: int = 0,
    subset: float = 1.0,
) -> DataLoader:
    dataset = PretrainingDataset(tokenizer, seq_length, subset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_fn(tokenizer.pad_token_id),
        pin_memory=True,
    )


def _collate_fn(pad_token_id: int):
    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = (input_ids != pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    return collate
