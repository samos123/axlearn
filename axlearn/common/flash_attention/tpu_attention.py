# Copyright © 2023 Apple Inc.

"""Wrappers for FlashAttention on TPU in JAX with logit bias support."""
import functools
import logging
from typing import Optional

import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu import splash_attention as splash_attention_kernel
from jax.experimental.pallas.ops.tpu.flash_attention import BlockSizes as LegacyBlockSizes
from jax.experimental.pallas.ops.tpu.flash_attention import SegmentIds as LegacySegmentIds
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as _pallas_tpu_flash_attention,
)
from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds as SplashSegmentIds
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from axlearn.common.attention_bias import (
    BaseAttentionBias,
    CausalAttentionBias,
    MaskFnAttentionBias,
    SegmentIdAttentionBias,
    SlidingWindowAttentionBias,
    ZeroAttentionBias,
    split,
)
from axlearn.common.flash_attention.common import (
    BaseFlashAttention,
    get_segment_ids,
    repeat_kv_heads,
)
from axlearn.common.flash_attention.remat import FLASH_ATTN_RESIDUAL_NAME
from axlearn.common.kv_cache.base_kv_cache import BaseKVCache
from axlearn.common.utils import Nested, Tensor

MaskFnOrZero = MaskFnAttentionBias | ZeroAttentionBias


def _to_splash_mask(
    mask: MaskFnOrZero,
    *,
    mask_shape: tuple[int, int],
    q_seq_shards: int = 1,
) -> splash_attention_mask.Mask:
    """Converts a mask to a splash mask."""
    if not mask.has_value():
        return splash_attention_mask.FullMask(mask_shape)
    assert isinstance(mask, MaskFnAttentionBias)
    if isinstance(mask, CausalAttentionBias):
        return splash_attention_mask.CausalMask(shape=mask_shape, shard_count=q_seq_shards)
    elif isinstance(mask, SlidingWindowAttentionBias):
        left_size = mask.sliding_window_size
        return splash_attention_mask.LocalMask(
            shape=mask_shape, window_size=(left_size, 0), offset=0, shard_count=q_seq_shards
        )

    # Because mask.mask() may use jnp ops. e.g. jnp.logical_and.
    with jax.ensure_compile_time_eval():
        # This code is reached only when `kv_cache_type=None` (i.e., forward and prefill) and
        # `target_len == source_len` (i.e., self-attention) (see `check_tpu_splash_attention`).
        # `target_positions` and `source_positions` are always in the range [0, seq_len].
        target_positions = np.arange(mask_shape[0])[None, :, None]
        source_positions = np.arange(mask_shape[1])[None, None, :]
        # `mask.mask` expects rank 3 tensors.
        mask_array = np.asarray(mask.mask(target_positions, source_positions))
        mask_array = np.squeeze(mask_array, axis=0)

    # NumpyMask is backed by a dense [target_len, source_len] numpy array.
    # May consume a large amount of host memory for long sequences at compile time.
    return splash_attention_mask.NumpyMask(array=mask_array)


class TPUFlashAttention(BaseFlashAttention):
    """Wraps the common checks for TPU attention implementations."""

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch=input_batch, kv_cache_type=kv_cache_type):
            return False
        block_size = self.cfg.tpu_block_size
        if not self._check_block_size(input_batch=input_batch, block_size=block_size):
            return False
        return True


class TPUSplashAttention(TPUFlashAttention):
    """Wraps SplashAttention.

    This kernel should be used for majority of the cases, except when
    1. explicit bias is used.
    2. head_dim is not a multiple of 128.

    In these two cases, we fallback to the legacy implementation.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._use_fused = True

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch, kv_cache_type=kv_cache_type):
            return False
        bias: BaseAttentionBias = input_batch["bias"]
        _, _, explicit_bias = split(bias, MaskFnAttentionBias, SegmentIdAttentionBias)
        query: Tensor = input_batch["query"]
        head_dim = query.shape[-1]

        if explicit_bias.has_value():
            return self._log_unsupported("explicit bias is not supported.")

        if head_dim % splash_attention_kernel.NUM_LANES != 0:
            return self._log_unsupported(
                f"{head_dim=} is not divisible by {splash_attention_kernel.NUM_LANES=}"
            )

        if (
            not self.get_backend_overrides("splash_use_fused_bwd_kernel", True)
            and self.cfg.dropout_rate > 0.0
        ):
            # TODO (bailin): Support dropout with non-fused bwd kernel.
            return self._log_unsupported("dropout with non-fused bwd kernel is not supported.")

        # If user doesn't specify splash_use_fused_bwd_kernel, we have some defaults
        # or heuristics to decide whether to use fused bwd kernel.
        if (
            not self.cfg.backend_overrides
            or "splash_use_fused_bwd_kernel" not in self.cfg.backend_overrides
        ):
            # When dropout is enabled, we always use the fused bwd kernel.
            if self.cfg.dropout_rate > 0.0:
                self._use_fused = True
            else:
                # Heuristic for sliding window attention.
                sliding, _ = split(bias, SlidingWindowAttentionBias)
                key: Tensor = input_batch["key"]
                kv_seq_len = key.shape[1]
                # TODO(c_lan): Support logit_sinks for non-fused bwd kernel.
                if sliding.has_value() and "logit_sinks" not in input_batch:
                    if kv_seq_len >= 16 * 1024 and kv_seq_len / sliding.sliding_window_size >= 4.0:
                        logging.info(
                            "Not using fused kernel for splash attention backward pass for better "
                            "performance, because sliding_window_size=%d << kv_seq_len=%d.",
                            sliding.sliding_window_size,
                            kv_seq_len,
                        )
                        self._use_fused = False
        else:
            self._use_fused = self.get_backend_overrides("splash_use_fused_bwd_kernel", True)

        return True

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """See `BaseFlashAttention.__call__`."""
        cfg = self.config
        bias: BaseAttentionBias = input_batch["bias"]
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        logit_sink: Optional[Tensor] = input_batch.get("logit_sink", None)
        prng_key = input_batch.get("prng_key", None)

        if cfg.dropout_rate > 0.0 and prng_key is None:
            raise ValueError(
                "TPU SplashAttention requires a prng_key to be provided when dropout is enabled."
            )

        mask, segment_ids, _ = split(bias, MaskFnAttentionBias, SegmentIdAttentionBias)
        segment_id_tensor = get_segment_ids(query=query, key=key, segment_ids=segment_ids)
        seg_ids = None
        if segment_id_tensor is not None:
            seg_ids = SplashSegmentIds(q=segment_id_tensor, kv=segment_id_tensor)

        query = query * self.cfg.softmax_scale
        # Switch num_heads and seq_len axes.
        query = jnp.einsum("btnh->bnth", query)
        key = jnp.einsum("bsnh->bnsh", key)
        value = jnp.einsum("bsnh->bnsh", value)

        block_size = self.cfg.tpu_block_size
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=self.get_backend_overrides("splash_block_q", block_size),
            block_kv=self.get_backend_overrides("splash_block_kv", block_size),
            block_kv_compute=self.get_backend_overrides("splash_block_kv_compute", block_size),
            # When fused bwd kernel is used, dq and dk/dv are computed in the same kernel. Only
            # *dkv* block sizes are used. When fused bwd kernel is not used, dk and dv are computed
            # in one kernel using *dkv* block sizes, and dq is computed in another kernel using *dq
            # block sizes.
            block_q_dkv=self.get_backend_overrides("splash_block_q_dkv", block_size),
            block_kv_dkv=self.get_backend_overrides("splash_block_kv_dkv", block_size),
            block_kv_dkv_compute=self.get_backend_overrides(
                "splash_block_kv_dkv_compute", block_size
            ),
            block_q_dq=None
            if self._use_fused
            else self.get_backend_overrides("splash_block_q_dq", block_size),
            block_kv_dq=None
            if self._use_fused
            else self.get_backend_overrides("splash_block_kv_dq", block_size),
            # The fused kernel is neutral in small models and a ~5%-15% improvement in larger ones.
            # E.g., 1.03x speedup in a 12.6b simulated model, 1.06x speedup in 29.6b ,
            # and 1.14x in 539.5b.
            # NOTE(hanzhi-zhou): Fused bwd kernel may require more memory usage because it needs to
            # keep a temporary unreduced dq tensor of shape (kv_seq_len // block_kv_dkv, *q.shape)
            # in HBM. If memory usage is a problem, consider increasing block_kv_dkv or disabling
            # fused kernel.
            use_fused_bwd_kernel=self._use_fused,
        )
        splash_mask = _to_splash_mask(
            mask, mask_shape=(query.shape[2], key.shape[2]), q_seq_shards=1
        )

        num_heads = query.shape[1]
        mha_mask = splash_attention_mask.MultiHeadMask(masks=[splash_mask] * num_heads)

        def kernel(q, k, v, segment_ids):
            q = jnp.einsum("nth->tnh", q)
            k = jnp.einsum("nsh->snh", k)
            v = jnp.einsum("nsh->snh", v)
            context = splash_attention_kernel.splash_attention_mha(
                q,
                k,
                v,
                segment_ids=segment_ids,
                mask=mha_mask,
                block_sizes=block_sizes,
                head_shards=1,
                q_seq_shards=1,
                dropout_rate=cfg.dropout_rate,
                interpret=self.cfg.interpret,
                prng_key=prng_key,
                logit_sink=logit_sink,
                residual_checkpoint_name=f"tpu_attention.{FLASH_ATTN_RESIDUAL_NAME}",
            )
            return jnp.einsum("tnh->nth", context)

        context = jax.vmap(kernel, axis_name="batch")(q=query, k=key, v=value, segment_ids=seg_ids)
        return jnp.einsum("bnth->btnh", context)

    def get_dropout_mask(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """Auxiliary function to get the dropout mask for debugging purposes.
        It will return a boolean dropout mask of shape [batch, num_heads, seq_len, seq_len].
        """
        cfg = self.config
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        prng_key = input_batch.get("prng_key", None)

        if cfg.dropout_rate > 0.0 and prng_key is None:
            raise ValueError(
                "TPU SplashAttention requires a prng_key to be provided when dropout is enabled."
            )

        # Switch num_heads and seq_len axes.
        query = jnp.einsum("btnh->bnth", query) * self.cfg.softmax_scale
        key = jnp.einsum("bsnh->bnsh", key)

        block_size = self.cfg.tpu_block_size
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=block_size,
            block_kv=block_size,
            block_kv_compute=block_size,
            block_q_dkv=block_size,
            block_kv_dkv=block_size,
            block_kv_dkv_compute=block_size,
            use_fused_bwd_kernel=True,
        )

        kernel = functools.partial(
            splash_attention_kernel.get_dropout_mask,
            prng_key=prng_key,
            block_sizes=block_sizes,
            dropout_rate=cfg.dropout_rate,
        )
        v_kernel = jax.vmap(kernel, axis_name="batch")
        dropout_mask = v_kernel(query, key)
        return dropout_mask


class LegacyTPUFlashAttention(TPUFlashAttention):
    """Wraps the legacy (deprecated) implementation of TPU attention."""

    def is_supported(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
        kv_cache_type: Optional[type[BaseKVCache]],
    ) -> bool:
        """See `BaseFlashAttention.is_supported`."""
        if not super().is_supported(input_batch, kv_cache_type=kv_cache_type):
            return False
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        if query.dtype != key.dtype:
            return self._log_unsupported(f"{query.dtype=} != {key.dtype=}")
        if self.cfg.dropout_rate != 0.0:
            return self._log_unsupported("dropout is not supported.")
        logit_sink = input_batch.get("logit_sink", None)
        if logit_sink is not None:
            return self._log_unsupported("LegacyTPUFlashAttention doesn't support logit sink.")
        return True

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        input_batch: Nested[Tensor | BaseAttentionBias],
    ) -> Tensor:
        """See `BaseFlashAttention.__call__`."""
        bias: BaseAttentionBias = input_batch["bias"]
        causal_mask, segment_ids, explicit_bias = split(
            bias, CausalAttentionBias, SegmentIdAttentionBias
        )
        query: Tensor = input_batch["query"]
        key: Tensor = input_batch["key"]
        value: Tensor = input_batch["value"]
        segment_id_tensor = get_segment_ids(query=query, key=key, segment_ids=segment_ids)
        seg_ids = None
        if segment_id_tensor is not None:
            seg_ids = LegacySegmentIds(q=segment_id_tensor, kv=segment_id_tensor)
        key = repeat_kv_heads(query.shape[2], key)
        value = repeat_kv_heads(query.shape[2], value)
        # Switch num_heads and seq_len axes.
        query = jnp.einsum("btnh->bnth", query) * self.cfg.softmax_scale
        key = jnp.einsum("bsnh->bnsh", key)
        value = jnp.einsum("bsnh->bnsh", value)

        block_size = self.cfg.tpu_block_size
        # TODO(tom_gunter): See if we can do better block-size tuning.
        block_sizes = LegacyBlockSizes(
            block_q=block_size,
            block_k_major=block_size,
            block_k=block_size,
            block_b=1,
            block_q_major_dkv=block_size,
            block_k_major_dkv=block_size,
            block_k_dkv=block_size,
            block_q_dkv=block_size,
            block_k_major_dq=block_size,
            block_k_dq=block_size,
            block_q_dq=block_size,
        )
        context = _pallas_tpu_flash_attention(
            query,
            key,
            value,
            ab=explicit_bias.value(),
            segment_ids=seg_ids,
            causal=causal_mask.has_value(),
            block_sizes=block_sizes,
            debug=self.cfg.interpret,
        )
        return jnp.einsum("bnth->btnh", context)
