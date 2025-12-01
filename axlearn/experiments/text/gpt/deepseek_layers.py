# Copyright 2024 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deepseek v3 layers."""

from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.attention import BaseQKVLinear
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.layers import Linear, RMSNorm, get_activation_fn
from axlearn.common.metrics import WeightedSummary
from axlearn.common.mixture_of_experts import (
    BaseGating,
    TransformerFeedForwardMoE,
    _compute_expert_capacity,
    _create_over_capacity_ratio_summary,
    _cum_sum,
    _router_z_loss,
)
from axlearn.common.module import Module
from axlearn.common.utils import Tensor


class MultiheadLatentAttention(BaseQKVLinear):
    """Multi-Head Latent Attention (MLA) layer.

    As described in Deepseek v3 paper.
    """

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures MultiheadLatentAttention."""

        # Latent dimension for query compression.
        q_lora_rank: Required[int] = REQUIRED
        # Latent dimension for key/value compression.
        kv_lora_rank: Required[int] = REQUIRED
        # Dimension for the decoupled RoPE part of the query/key.
        qk_rope_head_dim: Required[int] = REQUIRED
        # Dimension for the non-RoPE part of the query/key.
        qk_nope_head_dim: Required[int] = REQUIRED
        # Dimension for the value head.
        v_head_dim: Required[int] = REQUIRED

        # Layer for query compression (Down projection).
        q_down_proj: Linear.Config = Linear.default_config()
        # Layer for query up projection.
        q_up_proj: Linear.Config = Linear.default_config()
        # Layer for KV compression (Down projection).
        kv_down_proj: Linear.Config = Linear.default_config()
        # Layer for KV up projection (for both Key and Value).
        kv_up_proj: Linear.Config = Linear.default_config()
        # Layer for Key RoPE part.
        k_rope_proj: Linear.Config = Linear.default_config()

        # Normalization layers.
        q_norm: RMSNorm.Config = RMSNorm.default_config()
        kv_norm: RMSNorm.Config = RMSNorm.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        # Query compression.
        self._add_child("q_down_proj", cfg.q_down_proj.set(output_dim=cfg.q_lora_rank))
        self._add_child("q_norm", cfg.q_norm.set(input_dim=cfg.q_lora_rank))

        q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self._add_child(
            "q_up_proj",
            cfg.q_up_proj.set(input_dim=cfg.q_lora_rank, output_dim=cfg.num_heads * q_head_dim),
        )

        # KV compression.
        self._add_child("kv_down_proj", cfg.kv_down_proj.set(output_dim=cfg.kv_lora_rank))
        self._add_child("kv_norm", cfg.kv_norm.set(input_dim=cfg.kv_lora_rank))

        # KV up projection.
        self._add_child(
            "k_up_proj",
            cfg.kv_up_proj.set(
                input_dim=cfg.kv_lora_rank, output_dim=cfg.num_heads * cfg.qk_nope_head_dim
            ),
        )
        self._add_child(
            "v_up_proj",
            cfg.kv_up_proj.clone().set(
                input_dim=cfg.kv_lora_rank, output_dim=cfg.num_heads * cfg.v_head_dim
            ),
        )

        # K RoPE projection.
        self._add_child("k_rope_proj", cfg.k_rope_proj.set(output_dim=cfg.qk_rope_head_dim))

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[Tensor] = None,
        query_positions: Optional[Tensor] = None,
    ) -> BaseQKVLinear.Output:
        cfg = self.config
        # query is [batch, target_length, input_dim]
        # key/value are [batch, source_length, input_dim]

        # 1. Query path
        q_lora = self.q_down_proj(query)
        q_lora = self.q_norm(q_lora)
        q = self.q_up_proj(q_lora)
        q = q.reshape(*q.shape[:-1], cfg.num_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        # q_nope = q[..., : cfg.qk_nope_head_dim]
        # q_pe = q[..., cfg.qk_nope_head_dim :]
        # We return combined Q.

        # 2. KV path
        if key is None:
            key = query

        kv_compressed = self.kv_down_proj(key)
        kv_compressed = self.kv_norm(kv_compressed)

        k_nope = self.k_up_proj(kv_compressed)
        k_nope = k_nope.reshape(*k_nope.shape[:-1], cfg.num_heads, cfg.qk_nope_head_dim)

        v = self.v_up_proj(kv_compressed)
        v = v.reshape(*v.shape[:-1], cfg.num_heads, cfg.v_head_dim)

        k_rope = self.k_rope_proj(key)  # [batch, source_length, qk_rope_head_dim]
        k_rope = k_rope.unsqueeze(-2)  # [batch, source_length, 1, qk_rope_head_dim]
        k_rope = jnp.broadcast_to(k_rope, (*k_rope.shape[:-2], cfg.num_heads, cfg.qk_rope_head_dim))

        k = jnp.concatenate([k_nope, k_rope], axis=-1)

        return BaseQKVLinear.Output(query=q, key=k, value=v)


class DeepSeekGroupedTopKGating(BaseGating):
    """DeepSeek Grouped Top-K Gating."""

    @config_class
    class Config(BaseGating.Config):
        """Configures DeepSeekGroupedTopKGating."""

        # Number of groups.
        n_group: Required[int] = REQUIRED
        # Number of experts to select per group.
        topk_group: Required[int] = REQUIRED
        # Total number of experts to select.
        num_selected_experts: Required[int] = REQUIRED
        # Scaling factor for routed experts.
        routed_scaling_factor: float = 1.0
        # Regularization.
        router_z_loss_weight: float = 0.0
        # Token dropping capacity factor.
        train_capacity_factor: float = 2.0
        eval_capacity_factor: float = 2.0
        expert_capacity: Optional[int] = None

    def forward(self, logits: Tensor) -> BaseGating.Output:
        cfg = self.config
        # logits: [batch, num_experts]
        # In AXLearn MoE, logits is [outer_batch, num_groups, group_size, num_experts]
        # Let's assume standard AXLearn shape: OGSE

        # We need to flatten OGS to treat them as tokens for the DeepSeek logic,
        # OR apply DeepSeek logic per token (which is what we want).
        # The AXLearn MoE implementation applies gating on OGSE logits.
        # DeepSeek logic is per token.

        scores = jax.nn.sigmoid(logits)

        # Reshape to [..., n_group, experts_per_group]
        experts_per_group = cfg.num_experts // cfg.n_group
        scores_grouped = scores.reshape(*scores.shape[:-1], cfg.n_group, experts_per_group)

        # Select top 2 experts in each group to compute group score
        topk_in_group_scores, _ = jax.lax.top_k(scores_grouped, 2)
        group_scores = jnp.sum(topk_in_group_scores, axis=-1)  # [..., n_group]

        # Select topk groups
        _, group_idx = jax.lax.top_k(group_scores, cfg.topk_group)  # [..., topk_group]

        # Create a mask for selected groups
        group_mask = jnp.sum(jax.nn.one_hot(group_idx, cfg.n_group), axis=-2)  # [..., n_group]

        # Expand mask to experts
        expert_mask = jnp.repeat(group_mask[..., None], experts_per_group, axis=-1)
        expert_mask = expert_mask.reshape(*scores.shape)  # [..., num_experts]

        # Mask scores
        masked_scores = jnp.where(expert_mask > 0, scores, 0.0)

        # Select final topk experts
        topk_weights, topk_idx = jax.lax.top_k(masked_scores, cfg.num_selected_experts)

        # Normalize weights
        topk_weights = (
            cfg.routed_scaling_factor
            * topk_weights
            / (jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20)
        )

        # Compute expert capacity
        expert_capacity = _compute_expert_capacity(
            expert_capacity=cfg.expert_capacity,
            capacity_factor=(
                cfg.train_capacity_factor if self.is_training else cfg.eval_capacity_factor
            ),
            group_size=logits.shape[-2],
            num_experts=cfg.num_experts,
        )

        # Construct dispatch/combine tensors
        # We iterate over the k selected experts.

        over_capacity_list = []
        mask_count_all = jnp.zeros(logits.shape[:-1] + (cfg.num_experts,), dtype=jnp.int32)

        combine_tensor = jnp.zeros(
            (*logits.shape[:-1], cfg.num_experts, expert_capacity), dtype=jnp.float32
        )

        for i in range(cfg.num_selected_experts):
            idx = topk_idx[..., i]  # [..., ]
            w = topk_weights[..., i]  # [..., ]

            # Mask for this selection
            mask_i = jax.nn.one_hot(idx, cfg.num_experts, dtype=jnp.int32)  # [..., num_experts]

            # Position is cumsum of current mask + count of previous assignments
            # _cum_sum computes cumsum along axis -2 (group_size/S dim in OGSE)
            position_in_expert_i = _cum_sum(mask_i, exclusive=True, axis=-2) + jnp.expand_dims(
                mask_count_all, -2
            )

            over_capacity_list.append(
                _create_over_capacity_ratio_summary(
                    mask=mask_i,
                    position_in_expert=position_in_expert_i,
                    capacity=expert_capacity,
                )
            )

            # Filter capacity
            valid_mask = jnp.less(position_in_expert_i, expert_capacity).astype(mask_i.dtype)
            mask_i *= valid_mask

            # Update counts for next iteration
            # We need to sum along the S dimension (axis -2) to get the count for the whole group?
            # Wait, mask_count_all should track the count of tokens assigned to each expert SO FAR in the group.
            # _cum_sum gives position within the group.
            # So mask_count_all should be the total count of assignments in the group.
            # "mask_count_all += jnp.einsum("ogse->oge", masks_list[i])" in TopKGating.
            # Yes, sum over S (axis -2).
            mask_count_all += jnp.sum(mask_i, axis=-2)

            # Update combine tensor
            # b = one_hot(position)
            # a = gate * mask * one_hot(index)

            # Filter gate with valid mask (mask_i is already filtered)
            # We need mask_i_flat (sum over experts) to filter gate?
            # mask_i is [OGSE].
            # gate_i is w [OGS].

            # mask_i_flat = sum(mask_i, axis=-1) # [OGS]

            # But wait, w corresponds to idx.
            # So we can just use mask_i to place w into the correct expert slot.

            # a = w[..., None] * mask_i.astype(jnp.float32) # [OGSE]

            # b = one_hot(position, capacity) # [OGSEC]

            # combine += einsum("ogse,ogsc->ogsec", a, b)

            a = jnp.expand_dims(w, -1) * mask_i.astype(jnp.float32)  # [OGSE]
            b = jax.nn.one_hot(position_in_expert_i, expert_capacity, dtype=jnp.float32)  # [OGSEC]

            combine_tensor += jnp.einsum("ogse,ogsc->ogsec", a, b)

        # Add summaries
        self.add_summary("router_z_loss", WeightedSummary(_router_z_loss(logits), 1))
        for i, over_capacity_i in enumerate(over_capacity_list):
            self.add_summary(f"over_capacity_{i + 1}", WeightedSummary(over_capacity_i, 1))

        return BaseGating.Output(
            combine_tensor=combine_tensor,
            dispatch_tensor=combine_tensor.astype(bool),
            router_z_loss=_router_z_loss(logits),
            load_balance_loss=jnp.array(
                0.0, dtype=jnp.float32
            ),  # No load balance loss in DeepSeek v3?
            # DeepSeek v3 uses "Auxiliary-Loss-Free Load Balancing" strategy,
            # but we might want to add it if configured.
            # For now return 0.
        )


class DeepSeekFeedForwardMoE(TransformerFeedForwardMoE):
    """DeepSeek MoE with Shared and Routed Experts."""

    @config_class
    class Config(TransformerFeedForwardMoE.Config):
        """Configures DeepSeekFeedForwardMoE."""

        # Number of shared experts.
        n_shared_experts: int = 1
        # Shared expert hidden dim (usually same as routed experts or multiple).
        shared_expert_hidden_dim: Optional[int] = None

        # Override gating to default to DeepSeekGroupedTopKGating
        gating: DeepSeekGroupedTopKGating.Config = DeepSeekGroupedTopKGating.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        shared_dim = cfg.shared_expert_hidden_dim or (cfg.hidden_dim * cfg.n_shared_experts)

        self._add_child(
            "shared_gate_proj",
            Linear.default_config().set(input_dim=cfg.input_dim, output_dim=shared_dim, bias=False),
        )
        self._add_child(
            "shared_up_proj",
            Linear.default_config().set(input_dim=cfg.input_dim, output_dim=shared_dim, bias=False),
        )
        self._add_child(
            "shared_down_proj",
            Linear.default_config().set(input_dim=shared_dim, output_dim=cfg.input_dim, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config

        # 1. Routed Experts (Standard MoE path)
        routed_output = super().forward(inputs)

        # 2. Shared Experts
        gate = self.shared_gate_proj(inputs)
        up = self.shared_up_proj(inputs)

        # Use the same activation function as the routed experts
        activation_fn = get_activation_fn(cfg.activation)
        if isinstance(cfg.activation, tuple):
            # Handle GLU-like activations if tuple
            # But here we have explicit gate and up projections.
            # If cfg.activation is tuple (e.g. ("linear", "silu")), it implies GLU.
            # In super().forward, it handles tuple activation by splitting weights.
            # Here we manually created gate and up.
            # So we should just apply the non-linear part to the gate?
            # Deepseek example: ff_gate = silu(gate) * up.
            # So we should use silu (or whatever is the non-linear part).

            # If cfg.activation is "nn.silu", we apply it to gate?
            # Wait, standard FFN with GLU has 3 matrices: gate, up, down.
            # And output is down(act(gate) * up).
            # If activation is "nn.relu", it's usually down(relu(up)). (2 matrices)
            # But Deepseek v3 uses SwiGLU (3 matrices).

            # Let's assume SwiGLU structure for shared experts too.
            # If cfg.activation is a tuple, e.g. ("linear", "nn.silu"),
            # get_activation_fn returns a function that might not be what we want directly if we split it.

            # Let's just assume silu for now as per Deepseek paper/code, or try to infer.
            # The example code uses `jax.nn.silu`.
            act = jax.nn.silu(gate)
        else:
            act = activation_fn(gate)

        shared_output = self.shared_down_proj(act * up)

        return routed_output + shared_output
