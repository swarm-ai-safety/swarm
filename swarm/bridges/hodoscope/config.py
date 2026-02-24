"""Configuration for the Hodoscope bridge (compatible with hodoscope >=0.2)."""

from pathlib import Path
from typing import Literal, Optional

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
        default="gemini/gemini-2.5-flash",
        description="LLM model for trajectory summarization (litellm format)",
    )

    embedding_model: str = Field(
        default="gemini/gemini-embedding-001",
        description="Embedding model for trajectory vectorization (litellm format)",
    )

    max_workers: int = Field(
        default=10,
        description="Max concurrent workers for summarize/embed calls",
    )

    embed_dim: Optional[int] = Field(
        default=None,
        description="Override embedding dimensionality (None = model default)",
    )

    projection: str = Field(
        default="tsne",
        description="Dimensionality reduction method (tsne, umap, pca, trimap, pacmap)",
    )

    group_by: str = Field(
        default="agent_type",
        description="Default metadata field for grouping trajectories",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for hodoscope analyze (reproducibility)",
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
