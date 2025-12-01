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

"""Tests for deepseek_layers.py"""

import jax
from absl.testing import parameterized

from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.experiments.text.gpt.deepseek_layers import (
    DeepSeekFeedForwardMoE,
    MultiheadLatentAttention,
)


class MultiheadLatentAttentionTest(TestCase):
    def test_forward(self):
        batch_size = 2
        seq_len = 10
        input_dim = 32
        num_heads = 4

        cfg = MultiheadLatentAttention.default_config().set(name="test")
        cfg.query_dim = input_dim
        cfg.key_dim = input_dim
        cfg.value_dim = input_dim
        cfg.num_heads = num_heads
        cfg.per_head_dim = 16  # Ignored by MLA but required by BaseQKVLinear

        cfg.q_lora_rank = 16
        cfg.kv_lora_rank = 16
        cfg.qk_nope_head_dim = 8
        cfg.qk_rope_head_dim = 8
        cfg.v_head_dim = 16

        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(query=inputs, key=inputs, value=inputs),
        )

        # Check shapes
        # Query: [batch, seq_len, num_heads, qk_nope + qk_rope]
        self.assertEqual(outputs.query.shape, (batch_size, seq_len, num_heads, 16))
        # Key: [batch, seq_len, num_heads, qk_nope + qk_rope]
        self.assertEqual(outputs.key.shape, (batch_size, seq_len, num_heads, 16))
        # Value: [batch, seq_len, num_heads, v_head_dim]
        self.assertEqual(outputs.value.shape, (batch_size, seq_len, num_heads, 16))


class DeepSeekFeedForwardMoETest(TestCase):
    @parameterized.product(is_training=(True, False))
    def test_forward(self, is_training):
        batch_size = 2
        seq_len = 10
        input_dim = 32
        hidden_dim = 64
        num_experts = 8
        num_groups = 2

        cfg = DeepSeekFeedForwardMoE.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.hidden_dim = hidden_dim
        cfg.num_experts = num_experts
        cfg.num_groups = num_groups
        cfg.n_shared_experts = 1

        cfg.gating.n_group = num_groups
        cfg.gating.topk_group = 1
        cfg.gating.num_selected_experts = 2

        layer = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))

        outputs, _ = F(
            layer,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )

        self.assertEqual(outputs.shape, (batch_size, seq_len, input_dim))
