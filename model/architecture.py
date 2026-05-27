"""MoE Transformer architecture targeting ~38B total / ~2.6B activated params.

Decoder-only MoE with grouped-query attention, RoPE, shared+ routed experts.
Designed for SWE/agentic task performance matching Qwen3.6-A3B-35B.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


class MoEConfig(PretrainedConfig):
    model_type = "swe_agent_moe"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=24,
        num_attention_heads=20,
        num_kv_heads=4,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=65536,
        rope_theta=10000000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        num_experts=32,
        num_experts_per_tok=1,
        num_shared_experts=1,
        expert_intermediate_size=9216,
        moe_layer_frequency=1,
        use_moe=True,
        shared_expert_intermediate_size=9216,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        expert_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        moe_aux_loss_coeff=0.01,
        moe_z_loss_coeff=0.001,
        eos_token_id=151643,
        pad_token_id=151643,
        bos_token_id=151643,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.expert_intermediate_size = expert_intermediate_size
        self.moe_layer_frequency = moe_layer_frequency
        self.use_moe = use_moe
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.expert_dropout = expert_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0, scaling_factor=1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32) / scaling_factor
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    x_rot = torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    return x * cos + x_rot * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin, position_ids)
        k = apply_rotary_emb(k, cos, sin, position_ids)

        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        mask = attention_mask
        if mask is not None and mask.dim() == 4:
            pass
        elif mask is not None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
            )
            mask = causal_mask[None, None, :seq_len, :seq_len]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_weights = attn_weights + mask[:, :, :seq_len, :seq_len]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, config: MoEConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class Top1Router(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        weights, indices = torch.topk(logits, self.num_experts_per_tok, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(logits.dtype)
        return indices, weights, logits


class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.z_loss_coeff = config.moe_z_loss_coeff

        self.router = Top1Router(config)
        self.shared_expert = FeedForward(config, config.shared_expert_intermediate_size)
        self.experts = nn.ModuleList([
            FeedForward(config, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        indices, weights, router_logits = self.router(x_flat)

        shared_out = self.shared_expert(x)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts_per_tok):
            expert_idx = indices[..., i]
            weight = weights[..., i:i+1]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    routed_out[mask] += weight[mask] * self.experts[e](x_flat[mask])

        aux_loss = self._compute_aux_loss(router_logits, indices)
        z_loss = self._compute_z_loss(router_logits)

        routed_out = routed_out.view(batch_size, seq_len, hidden_size)
        output = shared_out + routed_out

        return output, aux_loss, z_loss

    def _compute_aux_loss(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        num_tokens = router_logits.shape[0]
        probs = F.softmax(router_logits, dim=-1)
        counts = torch.zeros(self.num_experts, device=router_logits.device)
        for i in range(self.num_experts_per_tok):
            counts.scatter_add_(0, indices[..., i].flatten(), torch.ones(num_tokens, device=router_logits.device))
        frac_per_token = counts / num_tokens
        prob_per_expert = probs.mean(dim=0)
        aux_loss = self.num_experts * (frac_per_token * prob_per_expert).sum()
        return self.aux_loss_coeff * aux_loss

    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = log_z.pow(2).mean()
        return self.z_loss_coeff * z_loss


class TransformerBlock(nn.Module):
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        use_moe = config.use_moe and (layer_idx % config.moe_layer_frequency == 0)
        if use_moe:
            self.moe = MoELayer(config)
        else:
            self.moe = FeedForward(config)
        self.use_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, position_ids, attention_mask, past_key_value, use_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        if self.use_moe:
            x, aux_loss, z_loss = self.moe(x)
        else:
            x = self.moe(x)
            aux_loss = z_loss = torch.tensor(0.0, device=x.device)
        x = residual + x

        return x, aux_loss, z_loss


class MoEPreTrainedModel(PreTrainedModel):
    config_class = MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class MoEModel(MoEPreTrainedModel):
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        cos, sin = precompute_rope_freqs(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_router_logits: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.embed_tokens(input_ids)

        cos = self.rope_cos
        sin = self.rope_sin

        all_aux_loss = []
        all_z_loss = []

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x_out, aux_loss, z_loss = self._gradient_checkpointing_func(
                    layer.__call__, x, cos, sin, position_ids, attention_mask, past_key_values, use_cache
                )
            else:
                x_out, aux_loss, z_loss = layer(
                    x, cos, sin, position_ids, attention_mask, past_key_values, use_cache
                )
            x = x_out
            all_aux_loss.append(aux_loss)
            all_z_loss.append(z_loss)

        x = self.norm(x)
        return x, torch.stack(all_aux_loss).sum(), torch.stack(all_z_loss).sum()


class MoEForCausalLM(MoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.model = MoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_router_logits: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, aux_loss, z_loss = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache, output_router_logits
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + aux_loss + z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, past_length:]

        position_ids = torch.arange(
            past_length if past_key_values is not None else 0,
            past_length + input_ids.shape[1] if past_key_values is not None else input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
