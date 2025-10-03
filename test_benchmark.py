#!/usr/bin/env python3
"""
Standalone script to preload a model from GCS using Colocated Python.

This script reads the checkpoint index to determine the model structure and creates
appropriate TensorSpec objects for preloading.

Usage:
    python load_model_colocated.py --ckpt_path gs://your-bucket/path/to/checkpoint
"""

import argparse
import asyncio
import functools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
import pathwaysutils
from jax._src.mesh import thread_resources
from jax.experimental import colocated_python, mesh_utils
from jax.experimental.array_serialization import serialization as array_serialization
from jax.experimental.array_serialization import tensorstore_impl

from axlearn.common import utils
from axlearn.common.array_serialization import _async_deserialize
from axlearn.common.checkpointer import parse_step_from_dir, read_index_file
from axlearn.common.utils import TensorSpec, infer_mesh_shape


def check_all_devices_ready(timeout_secs=300, poll_interval=10):
    print("Checking if all Pathways devices are ready...")
    try:
        devices = jax.devices()
        if not devices:
            print("No devices found.")
            return False

        num_devices = len(devices)
        print(f"Found {num_devices} virtual devices. Attempting test computation...")

        # Create a mesh encompassing all devices
        mesh = jax.sharding.Mesh(
            mesh_utils.create_device_mesh((num_devices,)), axis_names=("data",)
        )

        # Define a simple sharded computation
        @jax.jit
        def check_fn(x):
            return jax.lax.pmean(x, axis_name="data")

        start_time = time.time()
        while time.time() - start_time < timeout_secs:
            try:
                # Create a sharded array
                a = jax.device_put(
                    jnp.arange(num_devices),
                    jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")),
                )
                # Run the computation
                result = check_fn(a)
                result.block_until_ready()
                print(f"Test computation successful. All {num_devices} devices appear ready.")
                return True

            # pylint: disable=broad-exception-caught
            except Exception as e:
                print(f"Test computation failed (devices might not be ready yet): {e}")
                print(f"Retrying in {poll_interval} seconds...")
                time.sleep(poll_interval)

        print(f"Timeout: Devices not ready within {timeout_secs} seconds.")
        return False

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error during device readiness check: {e}")
        return False


def _colocated_deserialize(
    shardings: Sequence[jax.sharding.NamedSharding],
    tensorstore_specs: Sequence[Dict[str, Any]],
    global_shapes: Sequence[tuple],
    dtypes: Sequence[jnp.dtype],
):
    # concurrent_bytes = 1099511627776
    concurrent_bytes = 34359738368 * 6  # multiple of 32GB
    cpu_devices = colocated_python.colocated_cpu_devices(jax.devices())
    print(f"{cpu_devices=}")

    if len(cpu_devices) > 1:
        cpu_mesh = colocated_python.colocated_cpu_devices(thread_resources.env.physical_mesh)
        print(f"{cpu_mesh=}")
        cpu_shardings = [
            jax.sharding.NamedSharding(cpu_mesh, sharding.spec) for sharding in shardings
        ]
        print(f"{cpu_shardings=}")
    else:
        cpu_shardings = [
            jax.sharding.SingleDeviceSharding(cpu_devices[0]) for sharding in shardings
        ]

    def output_spec_fn():
        return [
            jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)
            for shape, dtype, sharding in zip(global_shapes, dtypes, cpu_shardings)
        ]

    @colocated_python.colocated_python
    def run_deserializer():
        # Object should be created once per process.
        # pylint: disable=protected-access
        byte_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
        h2d_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
        thread_pool = ThreadPoolExecutor(1)

        future_arrays = jax.tree.map(
            functools.partial(
                _async_deserialize,
                byte_limiter=byte_limiter,
                h2d_limiter=h2d_limiter,
                single_thread_pool=thread_pool,
            ),
            cpu_shardings,
            tensorstore_specs,
            global_shapes,
            dtypes,
        )

        async def gather_func():
            return await asyncio.gather(*future_arrays)

        result = asyncio.run(gather_func())
        return result

    run_deserializer = run_deserializer.specialize(
        devices=cpu_devices,
        out_specs_fn=output_spec_fn,
    )

    # Try running in the current event loop if one exists, otherwise create new one
    result = run_deserializer()
    return result


def create_mesh(mesh_shape=(1, 1, 1, 1, 1, -1)):
    """Create a JAX mesh for distributed computation."""
    inferred_mesh_shape = infer_mesh_shape(mesh_shape)
    print(f"Using mesh shape {inferred_mesh_shape} for {len(jax.devices())} devices")
    devices = mesh_utils.create_device_mesh(inferred_mesh_shape)
    return jax.sharding.Mesh(devices, ("pipeline", "data", "expert", "fsdp", "seq", "model"))


def create_state_spec_from_checkpoint(ckpt_path: str):
    """Create a NestedTensorSpec from checkpoint index information."""
    index = read_index_file(ckpt_path)
    print(f"Read checkpoint index with {len(index)} entries")

    state_spec = {}

    for path, value in index:
        if path == "step":
            continue

        # Filter out learner state
        if is_learner_path(path):
            continue

        if isinstance(value, dict) and "shape" in value and "dtype" in value:
            # pylint: disable=eval-used
            shape = eval(value["shape"]) if isinstance(value["shape"], str) else value["shape"]
            dtype_str = value["dtype"]

            # Convert dtype string to jax dtype
            dtype = getattr(jnp, dtype_str, jnp.float32)
            if dtype == jnp.float32:
                dtype = jnp.bfloat16

            # Create nested dict structure from path
            keys = path.split("/")
            current = state_spec
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = TensorSpec(shape=shape, dtype=dtype)

    return state_spec


def is_learner_path(path: str) -> bool:
    """Check if a path is part of the learner state."""
    # Exclude all learner paths (optimizer state, ema, etc.)
    return path.startswith("learner/")


def get_inference_partition_spec(path: str, shape: tuple) -> jax.sharding.PartitionSpec:
    """Get inference-friendly partition spec based on tensor path and shape.

    Based on set_inference_partition_spec function logic:
    - Attention weights: shard on fsdp and model axes
    - Feed-forward linear1 weights: shard on (fsdp, model)
    - Feed-forward linear2 weights: shard on (model, fsdp)
    - Other parameters: shard on fsdp axis only
    """
    fsdp_axis = "fsdp"
    tp_axis = "model"

    # Check if this is an attention weight
    if any(attn_key in path for attn_key in ["self_attention", "cross_attention"]):
        if any(
            weight_key in path for weight_key in ["i_proj", "o_proj", "q_proj", "k_proj", "v_proj"]
        ):
            # Attention projection weights: shard on (fsdp, model)
            if len(shape) >= 2:
                return jax.sharding.PartitionSpec(fsdp_axis, tp_axis)

    # Check if this is a feed-forward layer weight
    elif "feed_forward" in path:
        if "linear1" in path and "weight" in path:
            # Feed-forward linear1 weights: shard on (fsdp, model)
            if len(shape) >= 2:
                return jax.sharding.PartitionSpec(fsdp_axis, tp_axis)
        elif "linear2" in path and "weight" in path:
            # Feed-forward linear2 weights: shard on (model, fsdp)
            if len(shape) >= 2:
                return jax.sharding.PartitionSpec(tp_axis, fsdp_axis)

    # For small 1D tensors, no sharding
    if len(shape) == 1 and shape[0] < 16:
        return jax.sharding.PartitionSpec()

    # For other parameters (embeddings, layer norms, etc.), shard on fsdp axis
    if len(shape) >= 1:
        return jax.sharding.PartitionSpec(tp_axis)

    # For scalars or unknown cases, no sharding
    return jax.sharding.PartitionSpec()


def create_checkpoint_spec_from_state(ckpt_dir: str, state_spec: dict):
    """Create checkpoint spec following the pattern from TensorStoreStateStorage._get_spec."""

    tensorstore_specs = []
    shapes = []
    dtypes = []
    shardings = []

    # Get current mesh for creating shardings
    mesh = thread_resources.env.physical_mesh
    if not mesh.shape:
        raise RuntimeError("Checkpoint restoration must take place within the context of a Mesh")

    # Process each tensor in the state spec
    for path, value in utils.flatten_items(state_spec, separator="/"):
        if isinstance(value, TensorSpec):
            # Get dtype
            dtype = getattr(value.dtype, "dtype", value.dtype)

            # Create storage path and tensorstore spec
            gda_path = os.path.join(ckpt_dir, "gda", path)
            tensorstore_spec = array_serialization.get_tensorstore_spec(gda_path)

            # Get inference-friendly partition spec based on tensor path and shape
            partition_spec = get_inference_partition_spec(path, value.shape)
            # model_axis_size = mesh.shape.get("model", 1)
            # # Replicate small 1D tensors that cannot be sharded.
            # if len(value.shape) == 1 and value.shape[0] < model_axis_size:
            #     partition_spec = jax.sharding.PartitionSpec()
            # else:
            #     partition_spec = jax.sharding.PartitionSpec("model")

            # Create sharding with the appropriate partition spec
            sharding = jax.sharding.NamedSharding(mesh, partition_spec)

            tensorstore_specs.append(tensorstore_spec)
            shapes.append(value.shape)
            dtypes.append(dtype)
            shardings.append(sharding)

    return tensorstore_specs, shardings, shapes, dtypes


def _default_deserialize(
    shardings: Sequence[jax.sharding.NamedSharding],
    tensorstore_specs: Sequence[Dict[str, Any]],
    global_shapes: Sequence[tuple],
    dtypes: Sequence[jnp.dtype],
):
    # concurrent_bytes = 1099511627776
    concurrent_bytes = 34359738368 * 6  # multiple of 32GB
    # Object should be created once per process.
    # pylint: disable=protected-access
    byte_limiter = tensorstore_impl._LimitInFlightBytes(concurrent_bytes)
    h2d_limiter = tensorstore_impl._LimitInFlightBytes(34359738368)
    thread_pool = ThreadPoolExecutor(1)

    future_arrays = jax.tree.map(
        functools.partial(
            _async_deserialize,
            byte_limiter=byte_limiter,
            h2d_limiter=h2d_limiter,
            single_thread_pool=thread_pool,
        ),
        shardings,
        tensorstore_specs,
        global_shapes,
        dtypes,
    )

    async def gather_func():
        return await asyncio.gather(*future_arrays)

    result = asyncio.run(gather_func())
    return result


def load_model_default(ckpt_path: str):
    """Main function to preload a model from GCS checkpoint."""
    step = parse_step_from_dir(ckpt_path)
    print(f"Starting model preload from: {ckpt_path} (step {step})")

    if not ckpt_path.startswith("gs://"):
        raise ValueError(f"Only GCS paths (gs://) are supported, got: {ckpt_path}")

    with create_mesh():
        print("Reading checkpoint structure...")
        state_spec = create_state_spec_from_checkpoint(ckpt_path)

        print(f"Found {len(jax.tree_util.tree_leaves(state_spec))} tensors in checkpoint")

        tensorstore_specs, shardings, shapes, dtypes = create_checkpoint_spec_from_state(
            ckpt_path, state_spec
        )

        print("Preloading checkpoint to TPU memory...")
        start_time = time.perf_counter()

        restored_values = _default_deserialize(
            shardings=shardings,
            tensorstore_specs=tensorstore_specs,
            global_shapes=shapes,
            dtypes=dtypes,
        )

        preload_time = time.perf_counter() - start_time
        print(f"Preload completed in {preload_time:.2f} seconds")
        print(f"Preloaded {len(restored_values)} arrays")

        return restored_values


def load_model_colocated(ckpt_path: str):
    """Main function to preload a model from GCS checkpoint."""
    step = parse_step_from_dir(ckpt_path)
    print(f"Starting model preload from: {ckpt_path} (step {step})")

    if not ckpt_path.startswith("gs://"):
        raise ValueError(f"Only GCS paths (gs://) are supported, got: {ckpt_path}")

    with create_mesh():
        print("Reading checkpoint structure...")
        state_spec = create_state_spec_from_checkpoint(ckpt_path)

        print(f"Found {len(jax.tree_util.tree_leaves(state_spec))} tensors in checkpoint")

        tensorstore_specs, shardings, shapes, dtypes = create_checkpoint_spec_from_state(
            ckpt_path, state_spec
        )

        print("Preloading checkpoint to CPU memory...")
        start_time = time.perf_counter()

        preloaded_values = _colocated_deserialize(
            shardings=shardings,
            tensorstore_specs=tensorstore_specs,
            global_shapes=shapes,
            dtypes=dtypes,
        )

        preload_time = time.perf_counter() - start_time
        print(f"Preload completed in {preload_time:.2f} seconds")
        print(f"Preloaded {len(preloaded_values)} arrays")

        print("Transferring arrays to TPU...")
        start_time = time.perf_counter()

        restored_values = [jax.device_put(v, s) for v, s in zip(preloaded_values, shardings)]

        transfer_time = time.perf_counter() - start_time
        print(f"Transfer completed in {transfer_time:.2f} seconds")

        return restored_values


def main():
    parser = argparse.ArgumentParser(description="Preload model from GCS checkpoint")
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="GCS path to checkpoint directory (e.g., gs://bucket/path/to/checkpoint)",
    )
    args = parser.parse_args()

    if os.getenv("JAX_PLATFORMS") == "proxy":
        pathwaysutils.initialize()
    else:
        jax.distributed.initialize()

    print(f"JAX devices: {jax.devices()}")
    check_all_devices_ready()

    print("--- Running colocated benchmark ---")
    # Extract profile dir from ckpt_path. The profile dir should be gs://bucket/profiles/
    hostname = os.uname().nodename
    profile_dir = f"gs://{args.ckpt_path.split('/')[2]}/profiles/{hostname}"
    jax.profiler.start_trace(log_dir=profile_dir)
    start_colocated_time = time.perf_counter()
    loaded_values_colocated = load_model_colocated(ckpt_path=args.ckpt_path)
    for x in loaded_values_colocated:
        x.block_until_ready()
    print(f"✅ Successfully loaded model from {args.ckpt_path}")
    print(f"Deserialize took {time.perf_counter() - start_colocated_time:.2f} seconds")
    print(f"   Total parameters: {sum(x.size for x in loaded_values_colocated):,}")
    jax.profiler.stop_trace()

    # Exit early if on pathways
    if os.getenv("JAX_PLATFORMS") == "proxy":
        sys.exit(0)

    print("\n--- Running default benchmark ---")
    start_default_time = time.perf_counter()
    loaded_values_default = load_model_default(ckpt_path=args.ckpt_path)
    print(f"✅ Successfully loaded model from {args.ckpt_path}")
    print(f"Deserialize took {time.perf_counter() - start_default_time:.2f} seconds")
    print(f"   Total parameters: {sum(x.size for x in loaded_values_default):,}")


if __name__ == "__main__":
    main()
