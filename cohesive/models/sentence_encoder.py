"""
GTR sentence encoder (768-D) — same family as vec2text / e2t.py.

Used for contrastive training on triples and for corrector inputs/targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..constants import EMBED_DIM, GTR_MODEL_NAME
except ImportError:
    from constants import EMBED_DIM, GTR_MODEL_NAME


class GTRSentenceEncoder(nn.Module):
    """
    Thin wrapper around ``SentenceTransformer`` so training can backprop.

    Forward returns L2-normalised [B, EMBED_DIM] embeddings.
    """

    def __init__(self, model_name: str = GTR_MODEL_NAME, device: str | None = None):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.st = SentenceTransformer(model_name, device=device)
        self._dim = EMBED_DIM

    @property
    def embed_dim(self) -> int:
        return self._dim

    def _model_device(self) -> torch.device:
        """Robust device from the first parameter (works after .to())."""
        try:
            return next(self.st.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings → [B, D] normalised. Backprop-safe."""
        feat = self.st.tokenize(texts)
        dev = self._model_device()
        feat = {k: v.to(dev) for k, v in feat.items()}
        out = self.st.forward(feat)
        emb = out["sentence_embedding"]
        return F.normalize(emb.float(), dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """No-grad inference encode. Caller controls eval/train mode."""
        return self.forward(texts)
