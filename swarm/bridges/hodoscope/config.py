"""Configuration for the Hodoscope bridge."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class HodoscopeConfig(BaseModel):
    """Configuration for the Hodoscope trajectory analysis bridge.

    Parsed from the ``hodoscope:`` section of a scenario YAML file
    or constructed programmatically.
    """

    output_dir: Path = Field(
        default=Path("runs/hodoscope"),
        description="Directory for trajectory JSONs and .hodoscope.json output",
    )

    summarize_model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model for trajectory summarization",
    )

    embedding_model: str = Field(
        default="gemini/gemini-embedding-001",
        description="Embedding model for trajectory vectorization",
    )

    projection: str = Field(
        default="tsne",
        description="Dimensionality reduction method (tsne, umap, pca)",
    )

    group_by: str = Field(
        default="agent_type",
        description="Default metadata field for grouping trajectories",
    )

    trajectory_unit: Literal["agent_epoch", "agent_run", "interaction"] = Field(
        default="agent_epoch",
        description=(
            "How to group interactions into trajectories: "
            "agent_epoch (one trajectory per agent per epoch), "
            "agent_run (one trajectory per agent across all epochs), "
            "interaction (one trajectory per interaction)"
        ),
    )
