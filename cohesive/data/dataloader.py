"""Training data: HaluEval dialogue split (HF parquet)."""

from __future__ import annotations

import pandas as pd

HALUEVAL_DIALOGUE_PARQUET = (
    "hf://datasets/pminervini/HaluEval/dialogue/data-00000-of-00001.parquet"
)

REQUIRED_COLUMNS = (
    "dialogue_history",
    "right_response",
    "hallucinated_response",
)


def load_training_dataframe(max_rows: int | None = None) -> pd.DataFrame:
    """Load HaluEval dialogue rows with the schema expected by cohesive training."""
    df = pd.read_parquet(HALUEVAL_DIALOGUE_PARQUET)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"HaluEval parquet missing columns {missing}; have {list(df.columns)}"
        )
    df = df[list(REQUIRED_COLUMNS)].copy()
    if max_rows is not None:
        df = df.head(max_rows)
    return df
