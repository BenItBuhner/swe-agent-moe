"""Tests for data pipelines (batch construction, shapes, tokenization)."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pretraining_dataset import PretrainingDataset, _collate_fn
from data.midtraining_dataset import MidtrainingDataset
from data.sft_dataset import SFTDataset
from transformers import AutoTokenizer


def _get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_pretraining_dataset():
    tokenizer = _get_tokenizer()
    ds = PretrainingDataset(tokenizer, seq_length=512, subset=0.001)
    it = iter(ds)
    batch = [next(it) for _ in range(4)]
    assert len(batch) == 4
    for item in batch:
        assert item["input_ids"].shape[0] == 512
        assert item["labels"].shape[0] == 512
    print(f"Pretrain dataset OK: 4 samples, seq_len=512")


def test_midtraining_dataset():
    tokenizer = _get_tokenizer()
    ds = MidtrainingDataset(tokenizer, seq_length=512, subset=0.001)
    it = iter(ds)
    batch = [next(it) for _ in range(2)]
    assert len(batch) == 2
    for item in batch:
        assert item["input_ids"].shape[0] == 512
        assert item["labels"].shape[0] == 512
    print(f"Midtrain dataset OK: 2 samples, seq_len=512")


def test_sft_dataset():
    tokenizer = _get_tokenizer()
    ds = SFTDataset(tokenizer, seq_length=512, subset=0.001)
    it = iter(ds)
    batch = [next(it) for _ in range(2)]
    assert len(batch) == 2
    for item in batch:
        assert item["input_ids"].shape[0] == 512
        assert item["labels"].shape[0] == 512
    print(f"SFT dataset OK: 2 samples, seq_len=512")


def test_collate_fn():
    tokenizer = _get_tokenizer()
    collate = _collate_fn(tokenizer.pad_token_id)
    batch = [
        {"input_ids": torch.arange(512), "labels": torch.arange(512)},
        {"input_ids": torch.arange(512), "labels": torch.arange(512)},
    ]
    result = collate(batch)
    assert result["input_ids"].shape == (2, 512)
    assert result["labels"].shape == (2, 512)
    assert result["attention_mask"].shape == (2, 512)
    print(f"Collate OK: batch shape {result['input_ids'].shape}")


def test_labels_shift():
    """Verify labels are shifted correctly during training."""
    tokenizer = _get_tokenizer()
    ds = PretrainingDataset(tokenizer, seq_length=128, subset=0.001)
    it = iter(ds)
    sample = next(it)

    # labels should model next-token prediction
    assert torch.equal(sample["input_ids"][1:], sample["labels"][:-1])
    print(f"Labels shift OK: input_ids[1:] == labels[:-1]")


if __name__ == "__main__":
    test_pretraining_dataset()
    test_midtraining_dataset()
    test_sft_dataset()
    test_collate_fn()
    test_labels_shift()
    print("\nAll data tests passed!")
