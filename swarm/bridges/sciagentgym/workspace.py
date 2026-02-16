"""Workspace filesystem management for SciAgentGym environments.

Handles mounting, managing, and cleaning up isolated workspaces for
agent task execution.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages workspace filesystems for SciAgentGym environments.
    
    Each environment can have an isolated workspace for storing:
    - Intermediate computation results
    - Downloaded data files
    - Generated plots and visualizations
    - Tool execution artifacts
    """
    
    def __init__(
        self,
        base_path: Path,
        max_size_mb: int = 1000,
    ) -> None:
        """Initialize workspace manager.
        
        Args:
            base_path: Base directory for all workspaces.
            max_size_mb: Maximum size in MB for each workspace.
        """
        self.base_path = base_path
        self.max_size_mb = max_size_mb
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def mount_workspace(
        self,
        env_id: str,
        subdirs: Optional[list[str]] = None,
    ) -> Path:
        """Mount a new workspace for an environment (replaces mock).
        
        Args:
            env_id: Environment ID.
            subdirs: Optional list of subdirectories to create.
            
        Returns:
            Path to the mounted workspace.
        """
        workspace_path = self.base_path / env_id
        
        logger.info("Mounting workspace for env %s at %s", env_id, workspace_path)
        
        # Create workspace directory
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        if subdirs is None:
            subdirs = ["data", "outputs", "artifacts", "tmp"]
        
        for subdir in subdirs:
            (workspace_path / subdir).mkdir(exist_ok=True)
        
        # Set permissions (read/write for owner only)
        os.chmod(workspace_path, 0o700)
        
        logger.info("Workspace mounted successfully: %s", workspace_path)
        return workspace_path
    
    def get_workspace_path(self, env_id: str) -> Optional[Path]:
        """Get path to existing workspace.
        
        Args:
            env_id: Environment ID.
            
        Returns:
            Path to workspace if it exists, None otherwise.
        """
        workspace_path = self.base_path / env_id
        if workspace_path.exists():
            return workspace_path
        return None
    
    def get_workspace_size(self, env_id: str) -> int:
        """Get current size of workspace in bytes.
        
        Args:
            env_id: Environment ID.
            
        Returns:
            Workspace size in bytes, or 0 if workspace doesn't exist.
        """
        workspace_path = self.get_workspace_path(env_id)
        if not workspace_path:
            return 0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(workspace_path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                try:
                    total_size += filepath.stat().st_size
                except Exception as e:
                    logger.warning("Error getting size of %s: %s", filepath, e)
        
        return total_size
    
    def check_workspace_quota(self, env_id: str) -> bool:
        """Check if workspace is within size quota.
        
        Args:
            env_id: Environment ID.
            
        Returns:
            True if within quota, False otherwise.
        """
        size_bytes = self.get_workspace_size(env_id)
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > self.max_size_mb:
            logger.warning(
                "Workspace %s exceeds quota: %.2f MB / %d MB",
                env_id,
                size_mb,
                self.max_size_mb,
            )
            return False
        
        return True
    
    def cleanup_workspace(self, env_id: str) -> bool:
        """Cleanup and remove workspace (replaces mock).
        
        Args:
            env_id: Environment ID.
            
        Returns:
            True if cleanup successful, False otherwise.
        """
        workspace_path = self.get_workspace_path(env_id)
        if not workspace_path:
            logger.debug("Workspace %s does not exist, nothing to cleanup", env_id)
            return True
        
        logger.info("Cleaning up workspace: %s", workspace_path)
        
        try:
            shutil.rmtree(workspace_path)
            logger.info("Workspace cleaned up successfully: %s", workspace_path)
            return True
        except Exception as e:
            logger.error("Failed to cleanup workspace %s: %s", workspace_path, e)
            return False
    
    def cleanup_all_workspaces(self) -> None:
        """Cleanup all workspaces under base path."""
        logger.info("Cleaning up all workspaces under %s", self.base_path)
        
        if not self.base_path.exists():
            logger.debug("Base path does not exist, nothing to cleanup")
            return
        
        try:
            shutil.rmtree(self.base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info("All workspaces cleaned up")
        except Exception as e:
            logger.error("Failed to cleanup all workspaces: %s", e)
    
    def get_workspace_stats(self) -> dict:
        """Get statistics about all managed workspaces.
        
        Returns:
            Dictionary with workspace statistics.
        """
        if not self.base_path.exists():
            return {
                "total_workspaces": 0,
                "total_size_mb": 0,
                "workspaces": {},
            }
        
        workspaces = {}
        total_size = 0
        
        for workspace_dir in self.base_path.iterdir():
            if workspace_dir.is_dir():
                env_id = workspace_dir.name
                size_bytes = self.get_workspace_size(env_id)
                size_mb = size_bytes / (1024 * 1024)
                total_size += size_bytes
                
                workspaces[env_id] = {
                    "path": str(workspace_dir),
                    "size_mb": round(size_mb, 2),
                    "within_quota": size_mb <= self.max_size_mb,
                }
        
        return {
            "total_workspaces": len(workspaces),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_size_mb": self.max_size_mb,
            "workspaces": workspaces,
        }
