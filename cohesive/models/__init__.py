"""Model components. v4: GTR 768-D + HalluCorrector."""

from .hallucination_latent import CKPT_VERSION, EMBED_DIM

LATENT_DIM = EMBED_DIM
BGE_DIM = EMBED_DIM

__all__ = [
    "LATENT_DIM",
    "BGE_DIM",
    "CKPT_VERSION",
    "EMBED_DIM",
]
