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

"""Utilities to set up the Deepseek v3 GPT model trainer configs."""

from typing import Any, Dict

from axlearn.common.trainer_config_modifier import ChainConfigModifier, ModuleConfigModifier
from axlearn.experiments.text.gpt.common import mesh_shape_from_axes
from axlearn.experiments.text.gpt.deepseek_layers import (
    DeepSeekFeedForwardMoE,
    DeepSeekGroupedTopKGating,
    MultiheadLatentAttention,
)


def get_trainer_kwargs(
    model_size: str,
    *,
    vocab_size: int = 128 * 1024,
    flash_attention: bool = False,
) -> Dict[str, Any]:
    """Construct default trainer kwargs given a model size."""

    # Deepseek v3 specific configs
    # We use ModuleConfigModifier to swap in our custom layers.

    deepseek_modifiers = [
        # Replace QKVLinear with MultiheadLatentAttention
        ModuleConfigModifier.default_config().set(
            target_config="model.decoder.transformer.layer.self_attention.attention.input_linear",
            modification=MultiheadLatentAttention.default_config().set(
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
        ),
        # Replace FeedForward with DeepSeekFeedForwardMoE
        ModuleConfigModifier.default_config().set(
            target_config="model.decoder.transformer.layer.feed_forward",
            modification=DeepSeekFeedForwardMoE.default_config().set(
                num_experts=256,
                num_groups=8,
                n_shared_experts=1,
                gating=DeepSeekGroupedTopKGating.default_config().set(
                    n_group=8,
                    topk_group=4,
                    num_selected_experts=8,
                ),
            ),
        ),
    ]

    if model_size == "test":
        trainer_kwargs = dict(
            model_kwargs=dict(
                num_layers=2,
                hidden_dim=2048,  # 7168 in real model
                ffn_dim=2048,  # Ignored by MoE? No, used for shared expert default?
                # DeepSeekFeedForwardMoE uses hidden_dim for shared expert if not specified.
                num_heads=16,  # 128 in real model
                vocab_size=vocab_size,
                rope_theta=10000,
                flash_attention=flash_attention,
            ),
            learner_kwargs=dict(peak_lr=3e-4, weight_decay=0.1),
            max_sequence_length=1024,
            train_batch_size=4,
            max_step=100,
            mesh_shape=mesh_shape_from_axes(data=-1, fsdp=1),
            mesh_rules=(
                (
                    "tpu-v4-.*",
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            *deepseek_modifiers,
                        ]
                    ),
                ),
                (
                    "gpu-.*",  # Fallback for local testing
                    ChainConfigModifier.default_config().set(
                        config_modifiers=[
                            *deepseek_modifiers,
                        ]
                    ),
                ),
            ),
        )
    else:
        raise NotImplementedError(f"Model size {model_size} not implemented.")

    return trainer_kwargs
