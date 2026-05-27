"""Tests for the MoE model architecture, data pipelines, and RL environments."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.architecture import (
    MoEConfig, MoEForCausalLM, MoEModel,
    GroupedQueryAttention, RMSNorm, MoELayer,
    precompute_rope_freqs,
)
from configs.model_config import MoEModelConfig


def _make_tiny_config():
    return MoEConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        num_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )


def test_model_initialization():
    config = _make_tiny_config()
    model = MoEForCausalLM(config)
    assert model is not None
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0
    print(f"Model initialized: {num_params/1e6:.2f}M params")


def test_model_forward():
    config = _make_tiny_config()
    model = MoEForCausalLM(config)
    model.eval()

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs.logits is not None
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    print(f"Forward pass OK: logits shape {outputs.logits.shape}")


def test_model_loss():
    config = _make_tiny_config()
    model = MoEForCausalLM(config)
    model.train()

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs.loss is not None
    assert outputs.loss.item() > 0
    print(f"Loss computation OK: {outputs.loss.item():.4f}")


def test_moe_aux_loss():
    config = _make_tiny_config()
    model = MoEForCausalLM(config)

    hidden = torch.randn(2, 10, config.hidden_size)
    moe_layer = model.model.layers[0].moe

    assert hasattr(moe_layer, "router")
    output, aux_loss, z_loss = moe_layer(hidden)
    assert output.shape == hidden.shape
    print(f"MoE forward OK: output shape {output.shape}, aux_loss={aux_loss.item():.6f}, z_loss={z_loss.item():.6f}")


def test_grouped_query_attention():
    config = _make_tiny_config()
    attn = GroupedQueryAttention(config)

    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    cos, sin = precompute_rope_freqs(config.head_dim, seq_len, config.rope_theta)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    output = attn(x, cos, sin, position_ids)
    assert output.shape == x.shape
    kv_groups = config.num_attention_heads // config.num_kv_heads
    assert kv_groups == 2
    print(f"GQA forward OK: output shape {output.shape}, KV groups={kv_groups}")


def test_rms_norm():
    norm = RMSNorm(256, eps=1e-6)
    x = torch.randn(2, 10, 256)
    y = norm(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    print(f"RMSNorm OK")


def test_generation():
    config = _make_tiny_config()
    model = MoEForCausalLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        outputs = model.generate(
            prompt,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=config.pad_token_id,
        )
    assert outputs.shape[0] == 1
    assert outputs.shape[1] > 10
    print(f"Generation OK: output length {outputs.shape[1]}")


def test_config_params():
    cfg = MoEModelConfig()
    assert cfg.total_params > 0
    assert cfg.activated_params > 0
    assert cfg.activated_params < cfg.total_params
    print(f"Total params: {cfg.total_params:.2f}B, Activated: {cfg.activated_params:.2f}B")


def test_rope_precomputation():
    cos, sin = precompute_rope_freqs(128, 4096, 10000.0)
    assert cos.shape == (4096, 64)
    assert sin.shape == (4096, 64)
    assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-6)
    print(f"RoPE precomputation OK")


def test_moe_balance():
    config = _make_tiny_config()
    model = MoEForCausalLM(config)
    model.eval()

    x = torch.randn(4, 32, config.hidden_size)
    layer = model.model.layers[0]
    if hasattr(layer.moe, "router"):
        logits = layer.moe.router.gate(x.view(-1, config.hidden_size))
        probs = torch.softmax(logits, dim=-1)
        expert_usage = probs.mean(dim=0)
        print(f"Expert usage distribution: {expert_usage.detach().tolist()}")


if __name__ == "__main__":
    test_model_initialization()
    test_model_forward()
    test_model_loss()
    test_moe_aux_loss()
    test_grouped_query_attention()
    test_rms_norm()
    test_generation()
    test_config_params()
    test_rope_precomputation()
    test_moe_balance()
    print("\nAll tests passed!")
