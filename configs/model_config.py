"""Shared model and training configuration.

MoEModelConfig is the single source of truth for architecture hyperparameters.
TrainingConfig holds all training-phase hyperparameters and data mix definitions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class MoEModelConfig:
    """SWE-Agent MoE ~38B total / ~2.6B activated per token.

    Architecture targeting Qwen3-235M-A3B-35B equivalence for SWE/agent tasks.
    """

    # Architecture
    vocab_size: int = 152064
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 24
    num_attention_heads: int = 20
    num_kv_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    rope_theta: float = 10000000.0
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6

    # MoE
    num_experts: int = 32
    num_experts_per_tok: int = 1
    num_shared_experts: int = 1
    expert_intermediate_size: int = 9216
    moe_layer_frequency: int = 1
    use_moe: bool = True
    shared_expert_intermediate_size: int = 9216

    # Regularization
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    expert_dropout: float = 0.0

    # Embeddings
    tie_word_embeddings: bool = True

    # Init
    initializer_range: float = 0.02

    # Loss
    moe_aux_loss_coeff: float = 0.01
    moe_z_loss_coeff: float = 0.001

    # Generation
    eos_token_id: int = 151643
    pad_token_id: int = 151643
    bos_token_id: int = 151643

    @property
    def total_params(self) -> float:
        expert = 2 * self.hidden_size * self.expert_intermediate_size
        total_expert_params = (self.num_experts + self.num_shared_experts) * expert * self.num_hidden_layers
        attn = (4 * self.hidden_size * self.hidden_size) // self.num_attention_heads * (
            self.num_attention_heads + 2 * self.num_kv_heads
        )
        total_attn_params = attn * self.num_hidden_layers
        embed = self.vocab_size * self.hidden_size
        return (total_expert_params + total_attn_params + embed) / 1e9

    @property
    def activated_params(self) -> float:
        expert = 2 * self.hidden_size * self.expert_intermediate_size
        activated_expert = (self.num_experts_per_tok + self.num_shared_experts) * expert
        attn = (4 * self.hidden_size * self.hidden_size) // self.num_attention_heads * (
            self.num_attention_heads + 2 * self.num_kv_heads
        )
        per_layer = activated_expert + attn
        return (per_layer * self.num_hidden_layers) / 1e9

    def to_hf_config(self):
        """Convert to HuggingFace MoEConfig (model/architecture.py)."""
        from model.architecture import MoEConfig
        return MoEConfig(**{k: v for k, v in self.__dict__.items() if not k.startswith("_")})


@dataclass
class TrainingConfig:
    # Data
    pretrain_data_mix: List[str] = field(default_factory=lambda: [
        "codeparrot/github-code", "bigcode/the-stack-v2",
        "allenai/dolma", "cerebras/SlimPajama-627B",
        "HuggingFaceFW/fineweb-edu",
    ])
    midtrain_data_mix: List[str] = field(default_factory=lambda: [
        "Open-Orca/OpenOrca", "microsoft/orca-math",
        "camel-ai/code", "jeffdshen/reher-v0.1",
    ])
    sft_data_mix: List[str] = field(default_factory=lambda: [
        "DataClaw/AgentClaw", "DataClaw/CodeClaw",
        "DataClaw/SWEClaw", "DataClaw/ReasonClaw",
    ])

    # Pretraining
    pretrain_steps: int = 500_000
    pretrain_batch_size: int = 4
    pretrain_grad_accum: int = 8
    pretrain_seq_length: int = 4096
    pretrain_lr: float = 3e-4
    pretrain_warmup_steps: int = 2000
    pretrain_weight_decay: float = 0.1
    pretrain_optim: str = "adamw"
    pretrain_scheduler: str = "cosine"

    # Midtraining
    midtrain_steps: int = 50_000
    midtrain_batch_size: int = 4
    midtrain_grad_accum: int = 8
    midtrain_seq_length: int = 8192
    midtrain_lr: float = 1e-4
    midtrain_warmup_steps: int = 500

    # SFT
    sft_steps: int = 20_000
    sft_batch_size: int = 4
    sft_grad_accum: int = 8
    sft_seq_length: int = 8192
    sft_lr: float = 5e-5
    sft_warmup_steps: int = 200

    # RL
    rl_steps: int = 50_000
    rl_batch_size: int = 2
    rl_grad_accum: int = 4
    rl_seq_length: int = 16384
    rl_lr: float = 1e-6
    rl_kl_coeff: float = 0.04
    rl_gae_lambda: float = 0.95
    rl_clip_epsilon: float = 0.2

    # Checkpointing
    save_steps: int = 5000
    eval_steps: int = 1000
    output_dir: str = "outputs"
    logging_steps: int = 10

    # FSDP / DeepSpeed
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_auto_wrap_policy: str = "transformer_auto_wrap"
    fsdp_cpu_offload: bool = True
    fsdp_backward_prefetch: str = "backward_pre"
    fsdp_forward_prefetch: bool = True
    fsdp_use_orig_params: bool = False
    mixed_precision: str = "bf16"

    use_flash_attention: bool = True
    gradient_checkpointing: bool = True

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "swe-agent-moe"
