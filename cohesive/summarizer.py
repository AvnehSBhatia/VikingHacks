"""
BART summarisation only (no MMR compressor). Used on the hallucination branch:
``dialogue + bad response`` → short summary string.
"""

from __future__ import annotations

try:
    from .models.compressor import _bart_summarize, _default_embed_device
except ImportError:
    from cohesive.models.compressor import _bart_summarize, _default_embed_device


def summarize_hallucination_branch(
    dialogue_history: str,
    hallucinated_response: str,
    *,
    max_length: int = 150,
    min_length: int = 20,
    device: str | None = None,
) -> str:
    """
    Concatenate dialogue and the (false) model response, then BART-summarise.

    This summary is embedded and passed through the corrector; ground-truth uses
    the **full** correct prompt+answer embedding instead.
    """
    text = f"{dialogue_history.strip()}\n{hallucinated_response.strip()}".strip()
    if not text:
        return ""
    dev = device or _default_embed_device()
    return _bart_summarize(
        text,
        max_length=max_length,
        min_length=min_length,
        device=dev,
    )
