# pylint: disable=import-error
import argparse
import socket
import time

import jax
import numpy as np
from jax.experimental import multihost_utils

if __name__ == "__main__":
    jax.distributed.initialize()

    # Get hostname of the current process and gather them from all processes.
    hostname = socket.gethostname()

    # Pad hostname to a fixed length to use with all_gather.
    MAX_HOSTNAME_LEN = 256
    hostname_bytes = hostname.encode("utf-8")
    if len(hostname_bytes) > MAX_HOSTNAME_LEN:
        raise ValueError(f"Hostname '{hostname}' is too long (> {MAX_HOSTNAME_LEN} bytes)")

    padded_hostname = np.zeros(MAX_HOSTNAME_LEN, dtype=np.uint8)
    padded_hostname[: len(hostname_bytes)] = np.frombuffer(hostname_bytes, dtype=np.uint8)

    all_hostnames_padded = multihost_utils.process_allgather(padded_hostname)

    # Decode padded hostnames back to strings.
    all_hostnames = [h.tobytes().split(b"\x00", 1)[0].decode("utf-8") for h in all_hostnames_padded]

    all_devices = jax.devices()
    healthy_devices = multihost_utils.live_devices(all_devices)

    all_hosts = {d.process_index for d in all_devices}
    healthy_hosts = {d.process_index for d in healthy_devices}
    faulty_hosts = all_hosts - healthy_hosts

    # Map process indices to hostnames for reporting.
    all_hostnames_map = {i: name for i, name in enumerate(all_hostnames)}
    healthy_hostnames = {all_hostnames_map[i] for i in healthy_hosts}
    faulty_hostnames = {all_hostnames_map[i] for i in faulty_hosts}

    print(f"All hosts: {sorted(list(all_hostnames_map.values()))}")
    print(f"Healthy hosts: {sorted(list(healthy_hostnames))}")
    if faulty_hosts:
        print(f"FAULTY HOSTS: {sorted(list(faulty_hostnames))}")
    else:
        print("No faulty hosts detected.")
