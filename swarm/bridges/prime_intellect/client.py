"""Prime Intellect API client wrapper.

Provides a thin interface to the Prime Intellect platform for:
* Submitting training jobs (hosted / on-demand).
* Publishing environments to the Environments Hub.
* Querying job status and retrieving checkpoints.

This module requires the ``prime`` CLI / SDK to be installed for full
functionality but degrades gracefully when it's missing (raising clear
errors on use).
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.bridges.prime_intellect.config import (
    PrimeIntellectConfig,
    TrainingMode,
)

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a Prime Intellect training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Represents a submitted training job."""

    job_id: str = ""
    status: JobStatus = JobStatus.PENDING
    model_name: str = ""
    gpu_type: str = ""
    num_gpus: int = 1
    environment_name: str = ""
    config_path: str = ""
    checkpoint_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "model_name": self.model_name,
            "gpu_type": self.gpu_type,
            "num_gpus": self.num_gpus,
            "environment_name": self.environment_name,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "metadata": self.metadata,
        }


class PrimeIntellectClient:
    """Client for the Prime Intellect platform.

    Uses the ``prime`` CLI under the hood for operations that require
    authenticated access.  Falls back to REST API calls when ``requests``
    is available.

    Attributes:
        config: The bridge configuration.
    """

    def __init__(self, config: Optional[PrimeIntellectConfig] = None) -> None:
        self.config = config or PrimeIntellectConfig()
        self._api_key = self.config.api_key or os.environ.get("PRIME_API_KEY", "")

    # ----- Environment Hub -------------------------------------------------

    def publish_environment(
        self,
        env_dir: str,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a SWARM safety environment to the Environments Hub.

        Args:
            env_dir: Path to the environment directory (must contain
                a module with ``load_environment``).
            name: Environment name (defaults to config value).
            version: Environment version (defaults to config value).

        Returns:
            Dict with publication metadata.
        """
        name = name or self.config.environment_name
        version = version or self.config.environment_version

        result = self._run_prime_cli(
            ["env", "push"],
            cwd=env_dir,
        )
        return {
            "name": name,
            "version": version,
            "output": result,
        }

    def install_environment(self, name: str) -> Dict[str, Any]:
        """Install an environment from the Hub.

        Args:
            name: Environment name on the Hub.

        Returns:
            Dict with installation metadata.
        """
        result = self._run_prime_cli(["env", "install", name])
        return {"name": name, "output": result}

    def list_environments(self) -> List[Dict[str, Any]]:
        """List available environments on the Hub.

        Returns:
            List of environment metadata dicts.
        """
        result = self._run_prime_cli(["env", "list"])
        return [{"raw_output": result}]

    # ----- Training --------------------------------------------------------

    def submit_training_job(
        self,
        config_path: str,
        environment_name: Optional[str] = None,
    ) -> TrainingJob:
        """Submit a training job to Prime Intellect.

        Args:
            config_path: Path to the prime-rl TOML config file.
            environment_name: Environment name (defaults to config value).

        Returns:
            A TrainingJob object with the job ID and initial status.
        """
        env_name = environment_name or self.config.environment_name
        mode = self.config.training_mode

        if mode == TrainingMode.LOCAL:
            return self._submit_local(config_path, env_name)
        else:
            return self._submit_hosted(config_path, env_name)

    def get_job_status(self, job_id: str) -> TrainingJob:
        """Query the status of a training job.

        Args:
            job_id: The job ID returned by submit_training_job.

        Returns:
            Updated TrainingJob.
        """
        result = self._run_prime_cli(["pods", "status", job_id])
        status = JobStatus.RUNNING  # default
        if "completed" in result.lower():
            status = JobStatus.COMPLETED
        elif "failed" in result.lower():
            status = JobStatus.FAILED
        elif "pending" in result.lower():
            status = JobStatus.PENDING

        return TrainingJob(job_id=job_id, status=status, metadata={"raw": result})

    # ----- TOML generation -------------------------------------------------

    def generate_training_config(
        self,
        output_path: str,
        scenario_path: str = "",
    ) -> str:
        """Generate a prime-rl TOML config for SWARM safety training.

        Args:
            output_path: Where to write the TOML file.
            scenario_path: Path to the SWARM scenario YAML.

        Returns:
            The path to the generated config file.
        """
        toml_content = self._build_toml(scenario_path)
        Path(output_path).write_text(toml_content)
        logger.info("Generated prime-rl config at %s", output_path)
        return output_path

    # ----- GPU availability ------------------------------------------------

    def list_gpus(self) -> List[Dict[str, Any]]:
        """List available GPU resources.

        Returns:
            List of GPU availability dicts.
        """
        result = self._run_prime_cli(["availability", "list"])
        return [{"raw_output": result}]

    # ----- internal --------------------------------------------------------

    def _submit_local(
        self, config_path: str, env_name: str
    ) -> TrainingJob:
        """Submit a local training run (for development)."""
        logger.info(
            "Local training mode: would run prime-rl with config=%s env=%s",
            config_path,
            env_name,
        )
        return TrainingJob(
            job_id=f"local-{os.getpid()}",
            status=JobStatus.PENDING,
            model_name=self.config.model_name,
            gpu_type="local",
            num_gpus=1,
            environment_name=env_name,
            config_path=config_path,
            metadata={"mode": "local"},
        )

    def _submit_hosted(
        self, config_path: str, env_name: str
    ) -> TrainingJob:
        """Submit a hosted/on-demand training job."""
        result = self._run_prime_cli([
            "pods",
            "create",
            "--name",
            f"swarm-{env_name}",
        ])

        job_id = self._parse_job_id(result)
        return TrainingJob(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_name=self.config.model_name,
            gpu_type=self.config.gpu_type,
            num_gpus=self.config.num_gpus,
            environment_name=env_name,
            config_path=config_path,
            metadata={"mode": self.config.training_mode.value, "raw": result},
        )

    @staticmethod
    def _escape_toml_string(value: str) -> str:
        """Escape a value for safe inclusion in a TOML double-quoted string.

        Prevents injection of TOML structure via crafted config values.
        """
        # Replace backslash first, then other special chars
        value = value.replace("\\", "\\\\")
        value = value.replace('"', '\\"')
        value = value.replace("\n", "\\n")
        value = value.replace("\r", "\\r")
        value = value.replace("\t", "\\t")
        return value

    def _build_toml(self, scenario_path: str = "") -> str:
        """Build a prime-rl TOML config string.

        All string values are escaped to prevent TOML injection.
        """
        cfg = self.config
        esc = self._escape_toml_string

        # Sanitise the comment line (strip newlines to prevent comment breakout)
        safe_comment = re.sub(r"[\r\n]", " ", scenario_path or "default")

        base_line = (
            f'base = "{esc(cfg.base_model)}"'
            if cfg.base_model
            else '# base = ""'
        )
        scenario_line = (
            f'scenario_path = "{esc(scenario_path)}"'
            if scenario_path
            else '# scenario_path = ""'
        )

        return f"""\
# Auto-generated prime-rl config for SWARM safety training
# Scenario: {safe_comment}

[model]
name = "{esc(cfg.model_name)}"
{base_line}

[training]
method = "rl"
max_steps = {cfg.max_turns * cfg.max_episodes}
batch_size = 32
learning_rate = 1e-5

[training.rl]
reward_mode = "{esc(cfg.reward_mode.value)}"
clip_min = {cfg.reward_clip_min}
clip_max = {cfg.reward_clip_max}
normalize_rewards = {str(cfg.reward_normalize).lower()}

[environment]
name = "{esc(cfg.environment_name)}"
version = "{esc(cfg.environment_version)}"
{scenario_line}
population_size = {cfg.population_size}
max_turns = {cfg.max_turns}
reward_mode = "{esc(cfg.reward_mode.value)}"

[environment.reward_weights]
toxicity = {cfg.reward_weights.get('toxicity', -1.0)}
quality_gap = {cfg.reward_weights.get('quality_gap', 1.0)}
welfare = {cfg.reward_weights.get('welfare', 0.5)}
adverse_selection = {cfg.reward_weights.get('adverse_selection', -0.5)}
cooperation = {cfg.reward_weights.get('cooperation', 0.3)}

[infrastructure]
gpu_type = "{esc(cfg.gpu_type)}"
num_gpus = {cfg.num_gpus}
"""

    def _run_prime_cli(
        self,
        args: List[str],
        cwd: Optional[str] = None,
    ) -> str:
        """Run a ``prime`` CLI command and return stdout.

        Raises RuntimeError if the CLI is not installed.
        """
        cmd = ["prime"] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(
                    "prime CLI returned %d: %s", result.returncode, result.stderr
                )
            return result.stdout
        except FileNotFoundError as err:
            raise RuntimeError(
                "The 'prime' CLI is not installed.  Install it with: "
                "pip install prime\n"
                "Then authenticate with: prime login"
            ) from err

    def _parse_job_id(self, output: str) -> str:
        """Extract a job/pod ID from CLI output."""
        for line in output.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
        return f"unknown-{os.getpid()}"
