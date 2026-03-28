"""
Optional Tier B: hallucination corrector in GTR 768-D space (v4).

Load ``HalluCorrectorPipeline`` (alias ``HallucinationLatentSpace``) from a trained
``.pt`` checkpoint. Training: ``cohesive/training/train.py``. Inference helper:
``cohesive/conversation.py``.

Environment (optional): ``VIKING_HALLU_CHECKPOINT`` default path for Tier B.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

TIER_B_CHECKPOINT_ENV = "VIKING_HALLU_CHECKPOINT"


def default_hallu_checkpoint() -> Optional[str]:
    v = os.environ.get(TIER_B_CHECKPOINT_ENV)
    return v if v else None


def try_load_hallucination_latent_space(checkpoint_path: str | Path, device: str = "cpu") -> Any:
    """
    Load ``HalluCorrectorPipeline`` from a v4 checkpoint (``ckpt_version`` ≥ 4).
    """
    try:
        from cohesive.models.hallucination_latent import HallucinationLatentSpace  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Tier B requires cohesive.models.hallucination_latent.HallucinationLatentSpace."
        ) from e
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Hallucination checkpoint not found: {path.resolve()}")
    try:
        return HallucinationLatentSpace.load(str(path), device=device)
    except TypeError:
        return HallucinationLatentSpace.load(str(path))
