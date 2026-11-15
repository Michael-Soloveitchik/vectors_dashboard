"""Read-only registry that groups embeddings by model_name.

``SpeakerModel`` does NOT mutate the dataframe and does NOT perform
fusion.  Embedding-level fusion (``concat``, ``reinforce``, ``sum``,
``mean``) lives in the Audio project at
``modules_code/models/model_fused.py:FusedSpeakerModel`` and runs at
extraction time, not on the visualisation side.

This class is a thin accessor for the dashboard launcher notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SpeakerModel:
    """Thin read-only wrapper around the enriched dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: voice_id, session_id, device_id,
        model_name, centroid.
    """

    REQUIRED = ["voice_id", "session_id", "device_id", "model_name", "centroid"]

    def __init__(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"SpeakerModel: df missing columns {missing}")
        self.df = df

    def list_models(self) -> list[str]:
        """Sorted unique model names."""
        return sorted(self.df["model_name"].unique().tolist())

    def list_voices(self, model_name: str | None = None) -> list[str]:
        """Sorted unique voice IDs, optionally filtered to one model."""
        sub = self.df if model_name is None else self.df[self.df["model_name"] == model_name]
        return sorted(sub["voice_id"].unique().tolist())

    def filter_model(self, model_name: str) -> pd.DataFrame:
        """Return a copy filtered to a single model_name."""
        return self.df[self.df["model_name"] == model_name].reset_index(drop=True)

    def get_vectors(self, model_name: str) -> np.ndarray:
        """Stack all centroid vectors for a given model as (N, D)."""
        sub = self.filter_model(model_name)
        return np.stack(
            [np.asarray(c).reshape(-1) for c in sub["centroid"]], axis=0,
        )
