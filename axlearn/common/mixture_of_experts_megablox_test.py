# Copyright 2024 The JAX Authors.
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

"""Test for mixture_of_experts.py with Megablox backend."""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common import megablox
from axlearn.common.attention import TransformerFeedForwardLayer
from axlearn.common.mixture_of_experts import TransformerFeedForwardMoE
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, assert_allclose


class TransformerFeedForwardMoEMegabloxTest(TestCase):
    def setUp(self):
        super().setUp()
        if not hasattr(megablox, "gmm") or not megablox.backend:
            self.skipTest("Megablox backend not available")

    @parameterized.product(
        copy_weights=(True,),
        num_experts=(1, 4),
        activation=("nn.relu", ("nn.silu", "linear")),
    )
    def test_moe_and_dense_layer_parity(self, copy_weights, num_experts, activation):
        batch_size = 2
        seq_len = 128
        input_dim = 16
        hidden_dim = 64

        cfg = TransformerFeedForwardMoE.default_config().set(name="test")
        cfg.input_dim = input_dim
        cfg.hidden_dim = hidden_dim
        cfg.num_experts = num_experts
        cfg.activation = activation
        cfg.num_groups = 4
        cfg.backend = "megablox"
        # A large capacity factor to prevent dropping tokens.
        cfg.gating.train_capacity_factor = 100.0
        cfg.gating.eval_capacity_factor = 100.0
        layer: TransformerFeedForwardMoE = cfg.instantiate(parent=None)
        state = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        cfg_dense = TransformerFeedForwardLayer.default_config().set(name="dense_test")
        cfg_dense.input_dim = input_dim
        cfg_dense.hidden_dim = hidden_dim
        cfg_dense.activation = activation
        layer_dense: TransformerFeedForwardLayer = cfg_dense.instantiate(parent=None)
        state_dense = layer_dense.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )

        if copy_weights:
            # Copy the dense layer weights to all experts in the sparse model.
            if isinstance(activation, tuple):
                for i in range(len(activation)):
                    state[f"wi_{i}_weight"] = jnp.tile(
                        state_dense[f"linear1_{i}"]["weight"], [num_experts, 1, 1]
                    )
            else:
                state["wi_weight"] = jnp.tile(state_dense["linear1"]["weight"], [num_experts, 1, 1])
            state["wo_weight"] = jnp.tile(state_dense["linear2"]["weight"], [num_experts, 1, 1])

        inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=(batch_size, seq_len, input_dim))
        outputs, _ = F(
            layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=inputs),
        )
        outputs_dense, _ = F(
            layer_dense,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state_dense,
            inputs=dict(inputs=inputs),
        )
        # All experts in MoE have the same weights as the dense layers.
        # Since we use large capacity, outputs should match exactly (floating point errors).
        assert_allclose(outputs, outputs_dense, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
