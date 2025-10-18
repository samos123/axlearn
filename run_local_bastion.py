"""A script to run a fully functional AXLearn bastion service locally.

WARNING: This script is a full-featured orchestrator. It will connect to your
configured Kubernetes cluster and attempt to create real JobSet resources.
Ensure your kubectl is configured to point to a development/test cluster.
"""

# Copyright © 2023 Apple Inc.

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable

from absl import app, logging

# Key AXLearn cloud components.
from axlearn.cloud.common.bastion import Bastion, JobMetadata, new_jobspec, serialize_jobspec
from axlearn.cloud.common.cleaner import Cleaner
from axlearn.cloud.common.quota import QuotaFn, QuotaInfo
from axlearn.cloud.common.scheduler import JobScheduler
from axlearn.cloud.common.uploader import Uploader
from axlearn.cloud.common.utils import generate_job_id
from axlearn.common.config import config_class, config_for_function

# ----------------------------------------------------------------------------
# --- Functions to create and submit a test job to the local bastion. --------
# ----------------------------------------------------------------------------


def submit_test_job(active_jobs_dir: str):
    """Creates a JobSpec that runs a real GKE runner and places it in the active jobs directory."""
    job_name = f"local-bastion-test-{os.getpid()}"

    # This command instructs the bastion to run the real GKE runner.
    # The runner will in turn create a JobSet in your configured K8s cluster.
    job_command = (
        "python3 -m axlearn.cloud.gcp.jobs.launch run "
        "--runner_module=axlearn.cloud.gcp.runners "
        "--runner_name=gke_tpu_single "
        f"--name={job_name} "
        "--instance_type=tpu-v4-8 "
        "--bundler_spec=skip_bundle=True "  # Skip bundling for this simple job.
        "-- -- python3 -c 'import time; print(\"JobSet pod is running in K8s!\"); time.sleep(60); print(\"JobSet pod finished.')'"
    )

    print("\n" + "-" * 80)
    print(f"Submitting a test job named '{job_name}'")
    print(f"Command: {job_command}")

    metadata = JobMetadata(
        user_id=os.getenv("USER", "local-user"),
        project_id="my-gcp-project",  # Must match a project in your quota.json
        creation_time=datetime.now(timezone.utc),
        resources={"tpu-v4-8": 1},  # Request a TPU for scheduling purposes.
        priority=5,
        job_id=generate_job_id(),
    )
    jobspec = new_jobspec(name=job_name, command=job_command, metadata=metadata)

    job_spec_path = os.path.join(active_jobs_dir, job_name)
    with open(job_spec_path, "w", encoding="utf-8") as f:
        serialize_jobspec(jobspec, f)

    print(f"Successfully submitted job spec to: {job_spec_path}")
    print("You can monitor the JobSet with: kubectl get jobsets")
    print("-" * 80)


# ----------------------------------------------------------------------------
# --- Dummy components for the local bastion. --------------------------------
# ----------------------------------------------------------------------------


def noop_upload_fn_factory() -> Callable[[str, str], Any]:
    """Returns a callable that does nothing."""

    def fn(src: str, dst: str):
        logging.info("NO-OP UPLOADER: Skipping upload of %s to %s", src, dst)

    return fn


class NoOpUploader(Uploader):
    """An uploader that does nothing."""

    @config_class
    class Config(Uploader.Config):
        upload_fn: Callable[[str, str], Any] = config_for_function(noop_upload_fn_factory)

    def __call__(self):
        pass


class NoOpCleaner(Cleaner):
    """A cleaner that logs which jobs it would have cleaned, but does nothing."""

    def sweep(self, jobs):
        logging.info("NO-OP CLEANER: Pretending to clean jobs: %s", list(jobs.keys()))
        # In a real environment, this would check K8s and only return jobs
        # whose resources are gone. For local testing, we assume they are all gone
        # once they are completed, so the bastion can GC the job specs.
        return list(jobs.keys())


def quota_from_local_file(path: str) -> QuotaFn:
    """A QuotaFn that reads from a local JSON file."""

    def fn() -> QuotaInfo:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return QuotaInfo(
                total_resources=data.get("total_resources", []),
                project_resources=data.get("project_resources", {}),
                project_membership=data.get("project_membership", {}),
            )
        except (IOError, json.JSONDecodeError) as e:
            logging.warning("Could not read or parse quota file at %s: %s", path, e)
            return QuotaInfo(total_resources=[], project_resources={}, project_membership={})

    return fn


# ----------------------------------------------------------------------------
# --- Main bastion configuration and execution. ------------------------------
# ----------------------------------------------------------------------------


def configure_bastion_for_local() -> tuple[Bastion.Config, str]:
    """Configures the Bastion to run entirely on the local filesystem."""

    bastion_root_str = tempfile.mkdtemp(prefix="local-bastion-")
    bastion_root = f"file://{bastion_root_str}"

    active_jobs_dir = os.path.join(bastion_root_str, "jobs", "active")
    os.makedirs(active_jobs_dir, exist_ok=True)
    os.makedirs(os.path.join(bastion_root_str, "jobs", "states"), exist_ok=True)
    os.makedirs(os.path.join(bastion_root_str, "jobs", "complete"), exist_ok=True)
    os.makedirs(os.path.join(bastion_root_str, "jobs", "user_states"), exist_ok=True)
    os.makedirs(os.path.join(bastion_root_str, "logs"), exist_ok=True)

    print("-" * 80)
    print("Local Bastion Root Directory:")
    print(bastion_root_str)
    print("\nTo submit a job, copy your job spec JSON to:")
    print(active_jobs_dir)
    print("-" * 80)

    quota_file_path = os.path.expanduser("~/quota.json")
    if not os.path.exists(quota_file_path):
        logging.warning(
            "Quota file not found at %s. Creating a dummy file.",
            quota_file_path,
        )
        with open(quota_file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_resources": [{"tpu-v4-8": 8}, {"tpu-v4-8": 16}],
                    "project_resources": {"my-gcp-project": {"tpu-v4-8": 24}},
                    "project_membership": {"my-gcp-project": ["your-email@example.com"]},
                },
                f,
                indent=2,
            )

    job_scheduler_cfg = JobScheduler.default_config()

    bastion_cfg = Bastion.default_config().set(
        output_dir=bastion_root,
        scheduler=job_scheduler_cfg,
        quota=config_for_function(quota_from_local_file).set(path=quota_file_path),
        cleaner=NoOpCleaner.default_config(),
        uploader=NoOpUploader.default_config(),
        update_interval_seconds=15,
    )
    return bastion_cfg, active_jobs_dir


def main(_):
    """Main function to configure and run the local bastion."""
    print("--- AXLearn Local Bastion Runner ---")
    print(
        "\nWARNING: This script will attempt to create real resources in your configured K8s cluster."
    )
    print("\nPrerequisites:")
    print("  1. Docker is installed and running.")
    print("  2. You are authenticated to GCP (`gcloud auth application-default login`).")
    print("  3. Your `kubectl` is configured to point to a working Kubernetes cluster.")
    print("  4. A quota file exists at '~/quota.json'. A dummy file will be created if not found.")

    bastion_cfg, active_jobs_dir = configure_bastion_for_local()
    bastion = bastion_cfg.instantiate()

    submit_test_job(active_jobs_dir)

    print("\nStarting bastion loop. Press Ctrl+C to exit.")
    bastion.execute()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
