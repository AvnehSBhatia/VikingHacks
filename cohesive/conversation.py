"""
Online conversation manager (v4): summariser branch + GTR corrector pipeline.

``HalluCorrectorPipeline.process_turn(dialogue, bad_response)`` returns a 768-D vector
for vec2text or downstream use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .models.hallucination_latent import HalluCorrectorPipeline, TurnResult


@dataclass
class TurnRecord:
    turn_index: int
    dialogue_snippet: str
    summary_text: str
    hallucination_risk: float
    is_hallucination: bool
    timestamp: float = field(default_factory=time.time)


class ConversationSession:
    """Wraps ``HalluCorrectorPipeline`` for a single conversation."""

    def __init__(self, hls: HalluCorrectorPipeline):
        self.hls = hls
        self.history: list[TurnRecord] = []
        self._turn_idx = 0
        self._last_vec: torch.Tensor | None = None

    @classmethod
    def new(
        cls,
        model_path: str,
        device: str | None = None,
    ) -> "ConversationSession":
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        hls = HalluCorrectorPipeline.load(checkpoint_path=model_path, device=device)
        return cls(hls)

    def turn(
        self,
        dialogue_history: str,
        model_response: str,
        is_hallucination: bool = False,
    ) -> TurnResult:
        result = self.hls.process_turn(dialogue_history, model_response)
        self._last_vec = result.anti_hallucination_vector
        rec = TurnRecord(
            turn_index=self._turn_idx,
            dialogue_snippet=dialogue_history[-200:],
            summary_text=result.summary_text,
            hallucination_risk=result.hallucination_risk,
            is_hallucination=is_hallucination,
        )
        self.history.append(rec)
        self._turn_idx += 1
        return result

    def context_vector(self) -> torch.Tensor | None:
        return self._last_vec

    def save(self, path: str) -> None:
        state = {
            "hls_state": self.hls.state_dict(),
            "history": [vars(r) for r in self.history],
            "turn_idx": self._turn_idx,
        }
        torch.save(state, path)
        print(f"[Session] Saved to {path}")

    @classmethod
    def load(cls, session_path: str, model_path: str, device: str | None = None) -> "ConversationSession":
        sess = cls.new(model_path, device=device)
        state = torch.load(session_path, map_location="cpu", weights_only=False)
        if "hls_state" in state and "stretch_session" in state["hls_state"]:
            sess.hls.stretch.load_state_dict(state["hls_state"]["stretch_session"])
        sess._turn_idx = state.get("turn_idx", 0)
        sess.history = [TurnRecord(**r) for r in state.get("history", [])]
        return sess

    def summary(self) -> dict:
        if not self.history:
            return {"turns": 0}
        risks = [r.hallucination_risk for r in self.history]
        return {
            "turns": self._turn_idx,
            "mean_hallu_risk": round(sum(risks) / len(risks), 3),
        }
