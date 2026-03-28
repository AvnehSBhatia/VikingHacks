"""
Hallucination corrector pipeline (v4): GTR 768-D space + learnable S + 4-block residual.

Training (see cohesive.training.train):
  • Contrastive on triples (dialogue anchor, right positive, hallu negative)
  • Corrector: summary(hallu branch) → embed → S → residual → match embed(correct pair)

Inference: summarise bad branch → embed → corrector → 768-D vector for vec2text / injection.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..constants import CKPT_VERSION_V4, EMBED_DIM
except ImportError:
    from constants import CKPT_VERSION_V4, EMBED_DIM

from .hallu_corrector_module import HalluCorrectorModule
from .sentence_encoder import GTRSentenceEncoder

Vec = torch.Tensor
Mat = torch.Tensor

# Legacy names (tests / imports)
LATENT_DIM = EMBED_DIM
BGE_DIM = EMBED_DIM
CKPT_VERSION = CKPT_VERSION_V4


# ══════════════════════════════════════════════════════════════════════════════
# Online stretch (optional session state, 768-D)
# ══════════════════════════════════════════════════════════════════════════════


class StretchMatrix:
    """Identity + η·score·(v⊗axis); row-normalised. Separate from learnable S in module."""

    def __init__(self, dim: int = EMBED_DIM, eta: float = 0.02):
        self.dim = dim
        self.eta = eta
        self.S: Mat = torch.eye(dim)

    def update(self, v: Vec, axis: Vec, score: float) -> None:
        outer = torch.outer(v.reshape(-1), axis.reshape(-1))
        self.S = self.S + self.eta * score * outer
        norms = self.S.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.S = self.S / norms

    def apply(self, z: Vec) -> Vec:
        z = z.reshape(-1)
        return F.normalize(self.S @ z, dim=0)

    def reset(self) -> None:
        self.S = torch.eye(self.dim)

    def state_dict(self) -> dict:
        return {"S": self.S, "eta": self.eta, "dim": self.dim}

    def load_state_dict(self, d: dict) -> None:
        self.S = d["S"]
        self.eta = float(d.get("eta", 0.02))
        self.dim = int(d.get("dim", EMBED_DIM))


# ══════════════════════════════════════════════════════════════════════════════
# Result
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TurnResult:
    anti_hallucination_vector: Vec
    summary_text: str = ""
    hallucination_risk: float = 0.0

    @property
    def para_latent(self) -> Vec:
        """Compat: same as output vector."""
        return self.anti_hallucination_vector

    @property
    def para_latent_deformed(self) -> Vec:
        return self.anti_hallucination_vector

    @property
    def sentence_scores(self) -> list[float]:
        return []

    @property
    def log_det(self) -> float:
        return 0.0

    @property
    def gt_latent(self) -> Vec:
        return self.anti_hallucination_vector

    @property
    def anti_hallucination_text(self) -> str | None:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════


class HalluCorrectorPipeline(nn.Module):
    """
    GTR encoder + HalluCorrectorModule (learnable S + residual stack).

    ``process_turn(dialogue, bad_response)`` summarises the bad branch, embeds,
    runs corrector, returns 768-D vector.
    """

    def __init__(
        self,
        encoder: GTRSentenceEncoder,
        corrector: HalluCorrectorModule,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder = encoder
        self.corrector = corrector
        self.device = torch.device(device)
        self.encoder.to(device)
        self.corrector.to(device)
        self.stretch = StretchMatrix(dim=EMBED_DIM)

    def forward(self, z_summary: torch.Tensor) -> torch.Tensor:
        return self.corrector(z_summary)

    @torch.no_grad()
    def process_turn(self, dialogue_history: str, bad_response: str) -> TurnResult:
        try:
            from ..summarizer import summarize_hallucination_branch
        except ImportError:
            from summarizer import summarize_hallucination_branch

        dev = str(self.device)
        summary = summarize_hallucination_branch(
            dialogue_history, bad_response, device=dev
        )
        z_sum = self.encoder.encode_texts([summary]).to(self.device)
        z_hat = self.corrector(z_sum).squeeze(0)
        return TurnResult(
            anti_hallucination_vector=z_hat.cpu(),
            summary_text=summary,
        )

    def state_dict(self) -> dict:
        return {
            "ckpt_version": CKPT_VERSION_V4,
            "encoder": self.encoder.state_dict(),
            "corrector": self.corrector.state_dict(),
            "stretch_session": self.stretch.state_dict(),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"[HalluCorrector] Saved to {path} (ckpt_version={CKPT_VERSION_V4})")

    @classmethod
    def load(
        cls,
        path: str | None = None,
        device: str = "cpu",
        *,
        checkpoint_path: str | None = None,
    ) -> "HalluCorrectorPipeline":
        actual = path or checkpoint_path
        if actual is None:
            enc = GTRSentenceEncoder(device=device)
            cor = HalluCorrectorModule(EMBED_DIM)
            return cls(enc, cor, device=device)
        try:
            data = torch.load(actual, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(actual, map_location="cpu")
        ver = data.get("ckpt_version", 0)
        if ver < CKPT_VERSION_V4:
            raise ValueError(
                f"Checkpoint v{ver} is incompatible with v4 GTR corrector; train with cohesive/training/train.py"
            )
        enc = GTRSentenceEncoder(device=device)
        enc.load_state_dict(data["encoder"])
        cor = HalluCorrectorModule(EMBED_DIM)
        cor.load_state_dict(data["corrector"])
        pipe = cls(enc, cor, device=device)
        if "stretch_session" in data:
            pipe.stretch.load_state_dict(data["stretch_session"])
        return pipe

    def reset_conversation(self) -> None:
        self.stretch.reset()


# Alias for conversation.py and older scripts
HallucinationLatentSpace = HalluCorrectorPipeline
