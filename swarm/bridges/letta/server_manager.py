"""Manages Letta server lifecycle.

Supports three modes:
- ``external``: assumes server is already running, just health-checks
- ``docker``: starts a Docker container, polls until healthy
- ``local``: starts a local subprocess
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from typing import Any, Optional

from swarm.bridges.letta.config import LettaConfig

logger = logging.getLogger(__name__)


class LettaServerManager:
    """Manages Letta server lifecycle."""

    def __init__(self, config: LettaConfig) -> None:
        self.config = config
        self._process: Optional[Any] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start the Letta server based on server_mode.

        Returns True if the server is healthy after startup.
        """
        mode = self.config.server_mode

        if mode == "external":
            return self._start_external()
        elif mode == "docker":
            return self._start_docker()
        elif mode == "local":
            return self._start_local()
        else:
            raise ValueError(f"Unknown server_mode: {mode}")

    def _start_external(self) -> bool:
        """External mode: just health-check an already-running server."""
        logger.info(
            "Letta server mode=external, checking %s", self.config.base_url
        )
        if self.health_check():
            self._running = True
            return True
        logger.warning("Letta server at %s is not healthy", self.config.base_url)
        return False

    def _start_docker(self) -> bool:
        """Docker mode: start container and poll until healthy."""
        cmd = (
            f"docker run -d --rm -p {self.config.docker_port}:8283 "
            f"{self.config.docker_image}"
        )
        logger.info("Starting Letta docker container: %s", cmd)
        try:
            result = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("Docker start failed: %s", result.stderr)
                return False
            container_id = result.stdout.strip()
            self._process = container_id
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.error("Docker start failed: %s", exc)
            return False

        return self._poll_health()

    def _start_local(self) -> bool:
        """Local mode: start subprocess and poll until healthy."""
        cmd = "letta server --port 8283"
        logger.info("Starting Letta local server: %s", cmd)
        try:
            self._process = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("letta command not found")
            return False

        return self._poll_health()

    def _poll_health(self) -> bool:
        """Poll health check until timeout."""
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            if self.health_check():
                self._running = True
                logger.info("Letta server is healthy at %s", self.config.base_url)
                return True
            time.sleep(self.config.health_check_interval)
        logger.error(
            "Letta server startup timed out after %.1fs", self.config.startup_timeout
        )
        return False

    def health_check(self) -> bool:
        """Check if the Letta server is responding."""
        try:
            import urllib.request

            url = f"{self.config.base_url}/v1/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return bool(resp.status == 200)
        except Exception:
            return False

    def stop(self) -> None:
        """Stop the Letta server."""
        if self.config.server_mode == "docker" and isinstance(self._process, str):
            try:
                subprocess.run(
                    ["docker", "stop", self._process],
                    timeout=10,
                    capture_output=True,
                )
            except Exception:
                logger.warning("Failed to stop docker container %s", self._process)
        elif self.config.server_mode == "local" and self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None
        self._running = False
        logger.info("Letta server stopped")
