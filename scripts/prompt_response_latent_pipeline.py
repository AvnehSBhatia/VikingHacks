"""
Compress + local StretchMatrix (no cohesive checkpoint) + BART decode.

Run::

  python scripts/prompt_response_latent_pipeline.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cohesive.models.compressor import bart_embed_texts
from compressor import compress

LATENT_DIM = 1024


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class CompressedUnit:
    """BART encoder pooled vectors for one compressed turn."""

    paragraph_vector: torch.Tensor  # [1024]
    sentence_vectors: list[torch.Tensor]
    full_vector: torch.Tensor  # [1024] prompt+response


@dataclass
class TurnResult:
    anti_hallucination_vector: torch.Tensor  # [1024]
    sentence_scores: list[float]
    log_det: float
    paragraph_latent_deformed: torch.Tensor


class StretchMatrix1024:
    """Online deformation S ∈ R^{D×D}, starts at I. Same idea as cohesive StretchMatrix, D=1024."""

    def __init__(self, dim: int = LATENT_DIM, eta: float = 0.02):
        self.dim = dim
        self.eta = eta
        self.S: torch.Tensor = torch.eye(dim)

    def update(self, v: torch.Tensor, axis: torch.Tensor, score: float) -> None:
        v = F.normalize(v.detach().cpu(), dim=-1)
        axis = F.normalize(axis.detach().cpu(), dim=-1)
        outer = torch.outer(v, axis)
        self.S = self.S + self.eta * score * outer
        norms = self.S.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.S = self.S / norms

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        s = self.S.to(device=z.device, dtype=z.dtype)
        return F.normalize(s @ z, dim=-1)

    @property
    def log_abs_det(self) -> float:
        sign, logabsdet = torch.linalg.slogdet(self.S)
        if sign.item() <= 0:
            return logabsdet.item()
        return logabsdet.item()

    def reset(self) -> None:
        self.S = torch.eye(self.dim)


def _axis_from_unit(unit: CompressedUnit, device: torch.device) -> torch.Tensor:
    """Proxy hallucination axis without trained head: full vs paragraph spread."""
    p = F.normalize(unit.paragraph_vector.to(device), dim=-1)
    f = F.normalize(unit.full_vector.to(device), dim=-1)
    d = f - p
    if d.norm() < 1e-6:
        if unit.sentence_vectors:
            m = torch.stack([F.normalize(s.to(device), dim=-1) for s in unit.sentence_vectors]).mean(0)
            d = m - p
    if d.norm() < 1e-6:
        d = torch.randn(LATENT_DIM, device=device, dtype=torch.float32)
    return F.normalize(d, dim=-1)


def _build_compressed_unit(
    device: str,
    summary: str,
    prompt: str,
    response: str,
) -> CompressedUnit:
    full_text = f"{prompt.strip()}\n{response.strip()}"
    para = summary.strip() or full_text
    sents = _split_sentences(para)
    if not sents:
        sents = [para] if para else [full_text]

    para_vec = bart_embed_texts([para], device=device)[0].cpu()
    sent_vecs = [bart_embed_texts([s], device=device)[0].cpu() for s in sents]
    full_vec = bart_embed_texts([full_text], device=device)[0].cpu()
    return CompressedUnit(
        paragraph_vector=para_vec,
        sentence_vectors=sent_vecs,
        full_vector=full_vec,
    )


def process_turn_latent(
    unit: CompressedUnit,
    stretch: StretchMatrix1024,
    device: torch.device,
) -> TurnResult:
    """Deform paragraph/sentence latents through S; score sentences vs axis; blend to 1024-D."""
    axis = _axis_from_unit(unit, device)

    para_z = F.normalize(unit.paragraph_vector.to(device), dim=-1)
    sent_zs = [F.normalize(sv.to(device), dim=-1) for sv in unit.sentence_vectors]
    para_z_def = stretch.apply(para_z)

    sent_scores: list[float] = []
    sent_zs_def: list[torch.Tensor] = []
    eye = torch.eye(LATENT_DIM, device=device)
    for z in sent_zs:
        sc = (z * axis).sum().item()
        sent_scores.append(sc)
        if sc > 0:
            z_d = stretch.apply(z)
        else:
            s_inv = torch.linalg.solve(stretch.S.to(device), eye)
            z_d = F.normalize(s_inv @ z, dim=-1)
        sent_zs_def.append(z_d)

    if sent_zs_def:
        stacked = torch.stack(sent_zs_def)
        mean_sent = F.normalize(stacked.mean(dim=0), dim=-1)
        blended = F.normalize(0.5 * para_z_def + 0.5 * mean_sent, dim=-1)
    else:
        blended = para_z_def

    log_det = stretch.log_abs_det

    return TurnResult(
        anti_hallucination_vector=blended.cpu(),
        sentence_scores=sent_scores,
        log_det=log_det,
        paragraph_latent_deformed=para_z_def.cpu(),
    )


def _update_stretch_from_turn(
    unit: CompressedUnit,
    stretch: StretchMatrix1024,
    sentence_scores: list[float],
    device: torch.device,
) -> None:
    axis = _axis_from_unit(unit, device)
    for sv, sc in zip(unit.sentence_vectors, sentence_scores):
        stretch.update(sv.to(device), axis, float(sc))


def pairs_to_messages(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    """Each pair is (prompt, response) → user then assistant messages."""
    messages: list[dict] = []
    for prompt, response in pairs:
        messages.append({"role": "user", "content": str(prompt).strip()})
        messages.append({"role": "assistant", "content": str(response).strip()})
    return messages


@dataclass
class PipelineTurnOutput:
    turn_index: int
    prompt: str
    response: str
    compressed_summary: str
    sentence_scores: list[float]
    stretch_log_det: float
    dehallucinated_text: str
    anti_hallucination_vector: torch.Tensor


@dataclass
class LatentDeHallucinationPipeline:
    """
    BART encoder embeddings + conversation-level StretchMatrix; decode with BART decoder.
    Call ``reset_conversation()`` to clear S between sessions.
    """

    device: str | None = None
    stretch: StretchMatrix1024 = field(default_factory=StretchMatrix1024)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = _device()

    def reset_conversation(self) -> None:
        self.stretch.reset()

    def invert(self, vec: torch.Tensor | np.ndarray) -> str:
        from cohesive.models.compressor import bart_decode_from_vector

        t = torch.as_tensor(vec, dtype=torch.float32)
        if not torch.is_tensor(vec):
            t = t.float()
        return bart_decode_from_vector(t, device=self.device)

    def run_turns(
        self,
        pairs: Sequence[tuple[str, str]],
        max_tokens: int = 800,
        lambda_mmr: float = 0.5,
        update_stretch: bool = True,
    ) -> list[PipelineTurnOutput]:
        """
        For each prefix of pairs (turn 0..n-1), compress full history, run latent step, BART decode.
        StretchMatrix carries across turns in order.
        """
        dev_t = torch.device(self.device)
        out: list[PipelineTurnOutput] = []
        for i in range(len(pairs)):
            prefix = list(pairs[: i + 1])
            messages = pairs_to_messages(prefix)
            summary = compress(
                messages, max_tokens=max_tokens, lambda_mmr=lambda_mmr, device=self.device
            )
            prompt, response = pairs[i]
            unit = _build_compressed_unit(self.device, summary, prompt, response)
            turn = process_turn_latent(unit, self.stretch, dev_t)
            if update_stretch:
                _update_stretch_from_turn(unit, self.stretch, turn.sentence_scores, dev_t)
            text = self.invert(turn.anti_hallucination_vector)
            out.append(
                PipelineTurnOutput(
                    turn_index=i,
                    prompt=prompt,
                    response=response,
                    compressed_summary=summary,
                    sentence_scores=list(turn.sentence_scores),
                    stretch_log_det=turn.log_det,
                    dehallucinated_text=text,
                    anti_hallucination_vector=turn.anti_hallucination_vector,
                )
            )
        return out


def run_prompt_response_pipeline(
    pairs: Iterable[tuple[str, str]],
    *,
    reset_stretch_between_runs: bool = True,
    **kwargs,
) -> list[PipelineTurnOutput]:
    """Convenience: one-shot run over all pairs."""
    pairs_list = list(pairs)
    pipe = LatentDeHallucinationPipeline()
    if reset_stretch_between_runs:
        pipe.reset_conversation()
    return pipe.run_turns(pairs_list, **kwargs)


if __name__ == "__main__":
    demo_pairs: list[tuple[str, str]] = [
        ("What is Python?", "Python is a programming language."),
        ("Who created it?", "Guido van Rossum created Python in 1991."),
    ]
    results = run_prompt_response_pipeline(demo_pairs)
    for r in results:
        print("--- turn", r.turn_index, "---")
        print("summary:", r.compressed_summary[:200], "..." if len(r.compressed_summary) > 200 else "")
        print("scores:", [round(s, 4) for s in r.sentence_scores])
        print("log|det S|:", round(r.stretch_log_det, 6))
        print("decoded:", r.dehallucinated_text[:300])
