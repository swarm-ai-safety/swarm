"""Toolkit loading and registration for SciAgentGym.

Handles loading and registering scientific tools from SciAgentGym
toolkits across multiple disciplines.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_tools_for_disciplines(
    disciplines: List[str],
    sciagentgym_path: str,
) -> Dict[str, Any]:
    """Load tools from SciAgentGym for specified disciplines (live mode).
    
    This function replaces mock tool loading by actually importing and
    registering tools from the SciAgentGym toolkits directory.
    
    Args:
        disciplines: List of discipline names (e.g., ['physics', 'chemistry']).
        sciagentgym_path: Path to SciAgentGym installation root.
        
    Returns:
        Dictionary mapping tool names to tool classes.
        
    Raises:
        ImportError: If SciAgentGym is not properly installed.
        FileNotFoundError: If toolkit directory doesn't exist.
    """
    from gym.toolbox import Toolbox
    
    tools: Dict[str, Any] = {}
    sciagentgym_root = Path(sciagentgym_path)
    toolkits_path = sciagentgym_root / "toolkits"
    
    if not toolkits_path.exists():
        raise FileNotFoundError(
            f"SciAgentGym toolkits directory not found: {toolkits_path}"
        )
    
    logger.info(
        "Loading tools from SciAgentGym disciplines: %s", disciplines
    )
    
    for discipline in disciplines:
        discipline_path = toolkits_path / discipline
        if not discipline_path.exists():
            logger.warning(
                "Discipline directory not found: %s", discipline_path
            )
            continue
        
        discipline_tools = _load_discipline_tools(
            discipline, discipline_path, sciagentgym_root
        )
        tools.update(discipline_tools)
        
        logger.info(
            "Loaded %d tools from discipline '%s'",
            len(discipline_tools),
            discipline,
        )
    
    logger.info("Total tools loaded: %d", len(tools))
    return tools


def _load_discipline_tools(
    discipline: str,
    discipline_path: Path,
    sciagentgym_root: Path,
) -> Dict[str, Any]:
    """Load tools from a specific discipline directory.
    
    Args:
        discipline: Discipline name (e.g., 'physics').
        discipline_path: Path to discipline directory.
        sciagentgym_root: Root path of SciAgentGym installation.
        
    Returns:
        Dictionary mapping tool names to tool classes.
    """
    from gym.toolbox import Toolbox
    
    tools: Dict[str, Any] = {}
    
    # Find all subdirectories (topics) within the discipline
    for topic_dir in discipline_path.iterdir():
        if not topic_dir.is_dir():
            continue
        
        # Look for the tools_gym.py registration module
        tools_gym_file = topic_dir / f"{topic_dir.name}_tools_gym.py"
        
        if not tools_gym_file.exists():
            # Try alternative naming patterns
            alt_patterns = [
                topic_dir / f"{topic_dir.name}_gym.py",
                topic_dir / "tools_gym.py",
            ]
            for alt_file in alt_patterns:
                if alt_file.exists():
                    tools_gym_file = alt_file
                    break
        
        if tools_gym_file.exists():
            topic_tools = _load_tools_from_module(
                tools_gym_file, discipline, topic_dir.name, sciagentgym_root
            )
            tools.update(topic_tools)
    
    return tools


def _load_tools_from_module(
    module_path: Path,
    discipline: str,
    topic: str,
    sciagentgym_root: Path,
) -> Dict[str, Any]:
    """Load tools from a specific module file.
    
    Args:
        module_path: Path to the tools module file.
        discipline: Discipline name.
        topic: Topic name within the discipline.
        sciagentgym_root: Root path of SciAgentGym installation.
        
    Returns:
        Dictionary mapping tool names to tool classes.
    """
    from gym.toolbox import Toolbox
    
    tools: Dict[str, Any] = {}
    
    # Construct module import path
    relative_path = module_path.relative_to(sciagentgym_root)
    module_parts = list(relative_path.parts[:-1])  # Remove .py file
    module_parts.append(relative_path.stem)  # Add module name without .py
    module_name = ".".join(module_parts)
    
    try:
        # Import the module to trigger @Toolbox.register decorators
        module = importlib.import_module(module_name)
        
        # Get registered tools from Toolbox
        # Note: This assumes Toolbox has a registry we can access
        # The actual implementation depends on SciAgentGym's Toolbox API
        registered_tools = _get_registered_tools_from_module(module, topic)
        tools.update(registered_tools)
        
        logger.debug(
            "Loaded %d tools from %s/%s",
            len(registered_tools),
            discipline,
            topic,
        )
        
    except Exception as e:
        logger.warning(
            "Failed to load tools from %s: %s",
            module_path,
            e,
        )
    
    return tools


def _get_registered_tools_from_module(
    module: Any,
    topic: str,
) -> Dict[str, Any]:
    """Extract registered tools from an imported module.
    
    Args:
        module: Imported Python module.
        topic: Topic name for logging.
        
    Returns:
        Dictionary mapping tool names to tool classes.
    """
    from gym.tool import EnvironmentTool
    from gym.toolbox import Toolbox
    
    tools: Dict[str, Any] = {}
    
    # Scan module for classes decorated with @Toolbox.register
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        
        # Check if it's a class inheriting from EnvironmentTool
        if isinstance(attr, type) and issubclass(attr, EnvironmentTool):
            if hasattr(attr, "name"):
                tool_name = attr.name
                tools[tool_name] = attr
    
    return tools


def get_tool_registry_snapshot() -> Dict[str, Any]:
    """Get a snapshot of currently registered tools in Toolbox.
    
    Returns:
        Dictionary with tool registry information.
    """
    try:
        from gym.toolbox import Toolbox
        
        # This depends on Toolbox's internal API
        # May need adjustment based on actual implementation
        if hasattr(Toolbox, "_registry"):
            return dict(Toolbox._registry)
        else:
            logger.warning("Toolbox._registry not accessible")
            return {}
    except ImportError:
        logger.warning("SciAgentGym not installed, cannot get tool registry")
        return {}
