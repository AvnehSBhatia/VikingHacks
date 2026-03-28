"""
compressor.py
─────────────
Meta BART encoder–decoder for hallucination-aware context extension.

All semantic vectors are **BART encoder** mean-pooled hidden states (1024-D for
``facebook/bart-large``), L2-normalised — aligned with decoding via the same BART.

Objective: compress multi-turn dialogue, embed with BART, run the hallucination
latent stack, then **decode the output vector with the BART decoder** (no Ridge/T5).

Exports
───────
BGE_DIM / BART_EMBED_DIM       — 1024 (kept as BGE_DIM for latent checkpoint compat)
CompressedUnit                 — dataclass: text + BART encoder vectors
Compressor                     — class API for ConversationSession / training
compress                       — standalone: MMR + BART summarise (MMR uses BART embeds)
build_compressed_unit          — BART-embed a compressed string + full window
compressed_unit_from_paragraph — BART-embed sentences from a paragraph
bart_embed_text / bart_embed_texts
bart_decode_from_vector        — map [1024] → text via BART ``generate``

BART model : facebook/bart-large (Meta encoder–decoder)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

BART_EMBED_DIM = 1024
BGE_DIM = BART_EMBED_DIM  # alias: HallucinationLatentSpace checkpoints expect 1024-D
BART_MODEL_NAME = "facebook/bart-large"


# ══════════════════════════════════════════════════════════════════════════════
# CompressedUnit
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompressedUnit:
    """
    paragraph / sentences (text) + BART encoder pooled vectors [BART_EMBED_DIM].
    """
    paragraph:         str
    sentences:         list[str]
    paragraph_vector:  torch.Tensor        # [BART_EMBED_DIM]
    sentence_vectors:  list[torch.Tensor]  # each [BART_EMBED_DIM]
    full_vector:       torch.Tensor        # [BART_EMBED_DIM]
    compression_ratio: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def _stacked_embeddings_matrix(vecs: list[torch.Tensor]) -> np.ndarray:
    """
    Stack per-message encoder vectors to ``(n_messages, hidden_dim)``.

    If stacked layout is ``(hidden_dim, n_messages)`` (e.g. ``(1024, 2)`` for two
    messages), transpose so MMR rows and ``mean(axis=0)`` query share ``hidden_dim``.

    Does not load the HF model (tests patch ``_bart_embed_texts`` with small dims).
    """
    if not vecs:
        return np.zeros((0, 0), dtype=np.float64)
    stacked = torch.stack(vecs)
    n_msg = len(vecs)
    if stacked.ndim == 2:
        r, c = stacked.shape
        if r == n_msg:
            pass
        elif c == n_msg and r != n_msg:
            stacked = stacked.T
        else:
            raise ValueError(
                f"Cannot orient embeddings as ({n_msg}, *); got torch shape {tuple(stacked.shape)}"
            )
    else:
        stacked = stacked.reshape(n_msg, -1)
    mat = stacked.detach().cpu().numpy()
    if mat.shape[0] != n_msg:
        raise ValueError(
            f"Expected {n_msg} embedding rows after stack, got shape {mat.shape}"
        )
    return _normalize_rows(mat.astype(float))


def _default_embed_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# BART seq2seq singleton (encode + decode)
# ══════════════════════════════════════════════════════════════════════════════

_bart_bundle: dict[str, tuple] = {}


def _get_bart_seq2seq(device: str | None = None):
    """``(tokenizer, BartForConditionalGeneration)`` on ``device``."""
    if device is None:
        device = _default_embed_device()
    if device not in _bart_bundle:
        from transformers import AutoTokenizer, BartForConditionalGeneration

        tok = AutoTokenizer.from_pretrained(BART_MODEL_NAME)
        model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME)
        model = model.to(device).eval()
        _bart_bundle[device] = (tok, model)
    return _bart_bundle[device]


def _one_bart_pooled_row(pooled_i: torch.Tensor, hid: int) -> torch.Tensor:
    """
    Ensure one ``hid``-dim vector per message.

    Some HF / layout edge cases can yield extra leading dims; flatten then mean-pool
    any ``[k, hid]`` block down to ``[hid]``.
    """
    v = pooled_i.detach().float().reshape(-1)
    if v.numel() == hid:
        return v.cpu()
    if v.numel() % hid == 0:
        return v.reshape(-1, hid).mean(dim=0).cpu()
    raise ValueError(
        f"BART pooled row has length {v.numel()}, expected {hid} or a multiple of {hid}"
    )


def _bart_embed_texts(texts: list[str], device: str | None = None) -> list[torch.Tensor]:
    """Mean-pool last encoder hidden states; L2-normalise per row → CPU float tensors."""
    if not texts:
        return []
    if device is None:
        device = _default_embed_device()
    tok, model = _get_bart_seq2seq(device)
    enc = model.get_encoder()
    hid = int(getattr(model.config, "d_model", None) or BART_EMBED_DIM)
    with torch.no_grad():
        batch = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(1024, getattr(tok, "model_max_length", 1024) or 1024),
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
        pooled = F.normalize(pooled, dim=-1)
    return [_one_bart_pooled_row(pooled[i], hid) for i in range(len(texts))]


def bart_embed_texts(texts: list[str], device: str | None = None) -> list[torch.Tensor]:
    """Public: BART encoder embeddings for each string."""
    return _bart_embed_texts(texts, device=device)


def bart_embed_text(text: str, device: str | None = None) -> torch.Tensor:
    return bart_embed_texts([text], device=device)[0]


def bart_decode_from_vector(
    vec: torch.Tensor,
    *,
    device: str | None = None,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> str:
    """
    Decode a single 1024-D vector through the BART decoder (encoder hidden = one timestep).

    The vector is treated as ``last_hidden_state[:, :1, :]`` with attention mask 1.
    """
    from transformers.modeling_outputs import BaseModelOutput

    if device is None:
        device = _default_embed_device()
    tok, model = _get_bart_seq2seq(device)
    model.eval()
    if vec.dim() == 1:
        v = vec.to(device=device, dtype=model.dtype).unsqueeze(0).unsqueeze(0)
    elif vec.dim() == 2:
        v = vec.to(device=device, dtype=model.dtype).unsqueeze(1)
    else:
        raise ValueError(f"vec must be [D] or [1,D], got shape {tuple(vec.shape)}")
    attn = torch.ones(v.shape[:2], device=device, dtype=torch.long)
    enc_out = BaseModelOutput(last_hidden_state=v)
    start = model.config.decoder_start_token_id
    if start is None:
        start = tok.pad_token_id
    with torch.no_grad():
        ids = model.generate(
            encoder_outputs=enc_out,
            attention_mask=attn,
            decoder_start_token_id=start,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
    return tok.decode(ids[0], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# MMR
# ══════════════════════════════════════════════════════════════════════════════

def _mmr_select(
    embeddings: np.ndarray,
    query: np.ndarray,
    target_count: int,
    lambda_mmr: float,
) -> list[int]:
    n = len(embeddings)
    if n == 0:
        return []
    k = min(target_count, n)
    selected: list[int] = []
    remaining = list(range(n))
    for _ in range(k):
        best_idx: Optional[int] = None
        best_score = float("-inf")
        for i in remaining:
            rel = float(embeddings[i] @ query)
            red = max(
                (float(embeddings[i] @ embeddings[j]) for j in selected),
                default=0.0,
            )
            score = lambda_mmr * rel - (1.0 - lambda_mmr) * red
            if best_idx is None or score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)   # type: ignore[arg-type]
        remaining.remove(best_idx)  # type: ignore[arg-type]
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Lazy summarization (pipeline or generate)
# ══════════════════════════════════════════════════════════════════════════════

_summ_pipe_cache: dict[str, object] = {}


def _lazy_bart_summarizer():
    if "pipe" not in _summ_pipe_cache:
        try:
            from transformers import pipeline

            _summ_pipe_cache["pipe"] = pipeline("summarization", model=BART_MODEL_NAME)
            _summ_pipe_cache["use_pipe"] = True
        except (KeyError, Exception):
            _summ_pipe_cache["pipe"] = None
            _summ_pipe_cache["use_pipe"] = False
    return _summ_pipe_cache["pipe"], _summ_pipe_cache["use_pipe"]


def _bart_summarize(text: str, *, max_length: int, min_length: int, device: str | None = None) -> str:
    pipe, use_pipe = _lazy_bart_summarizer()
    if use_pipe and pipe is not None:
        out = pipe(
            text,
            max_length=max_length,
            min_length=min(min_length, max_length - 1),
            do_sample=False,
        )
        return str(out[0]["summary_text"]).strip()

    if device is None:
        device = _default_embed_device()
    tok, model = _get_bart_seq2seq(device)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min(min_length, max_length - 1),
            do_sample=False,
            num_beams=4,
        )
    return tok.decode(ids[0], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Standalone compress()
# ══════════════════════════════════════════════════════════════════════════════

_ROLE_LABEL = {"user": "User", "assistant": "Assistant", "system": "System"}


def compress(
    messages: list[dict],
    target_count: int | None = None,
    lambda_mmr: float = 0.5,
    *,
    max_tokens: int = 800,
    device: str | None = None,
) -> str:
    """
    MMR (BART encoder similarities) → BART summarization of selected messages.
    """
    if not 0.0 <= lambda_mmr <= 1.0:
        raise ValueError(f"lambda_mmr must be in [0, 1], got {lambda_mmr!r}")
    if not messages:
        return ""

    dev = device or _default_embed_device()
    contents = [m.get("content", "") for m in messages]

    embeddings = _stacked_embeddings_matrix(_bart_embed_texts(contents, device=dev))

    if target_count is None:
        tok, _ = _get_bart_seq2seq(dev)
        ml = getattr(tok, "model_max_length", 1024) or 1024
        try:
            ml = int(ml)
        except (TypeError, ValueError):
            ml = 1024
        ml = max(32, min(ml, 4096))
        token_counts = [len(tok.encode(c, truncation=True, max_length=ml)) for c in contents]
        budget, cnt = max_tokens, 0
        for tc in reversed(token_counts):
            if budget >= tc:
                budget -= tc
                cnt += 1
            else:
                break
        target_count = max(1, cnt)

    query = _normalize_rows(embeddings.mean(axis=0, keepdims=True))[0]
    sel_order = _mmr_select(embeddings, query, target_count, lambda_mmr)
    sel_sorted = sorted(sel_order)

    parts = []
    for i in sel_sorted:
        role_label = _ROLE_LABEL.get(messages[i].get("role", "").lower(), messages[i].get("role", ""))
        parts.append(f"{role_label}: {contents[i]}")
    bart_input = " ".join(parts)

    return _bart_summarize(
        bart_input,
        max_length=max_tokens,
        min_length=min(10, max_tokens),
        device=dev,
    )


# ══════════════════════════════════════════════════════════════════════════════
# build_compressed_unit / compressed_unit_from_paragraph
# ══════════════════════════════════════════════════════════════════════════════

def build_compressed_unit(
    compressed_text: str,
    full_text: str,
    *,
    device: str | None = None,
) -> CompressedUnit:
    text = compressed_text.strip()
    sentences = _split_sentences(text) or [text]

    dev = device or _default_embed_device()
    para_vecs = _bart_embed_texts([text], device=dev)
    sent_vecs = _bart_embed_texts(sentences, device=dev)
    full_vecs = _bart_embed_texts([full_text], device=dev)

    orig_words = max(1, len(full_text.split()))
    comp_words = max(1, len(text.split()))

    return CompressedUnit(
        paragraph=text,
        sentences=sentences,
        paragraph_vector=para_vecs[0],
        sentence_vectors=sent_vecs,
        full_vector=full_vecs[0],
        compression_ratio=orig_words / comp_words,
    )


def compressed_unit_from_paragraph(paragraph: str, *, device: str | None = None) -> CompressedUnit:
    text = paragraph.strip()
    if not text:
        raise ValueError("paragraph must be non-empty")

    dev = device or _default_embed_device()
    sentences = _split_sentences(text) or [text]
    para_v = _bart_embed_texts([text], device=dev)[0]
    sent_vs = _bart_embed_texts(sentences, device=dev)

    return CompressedUnit(
        paragraph=text,
        sentences=sentences,
        paragraph_vector=para_v,
        sentence_vectors=sent_vs,
        full_vector=para_v.clone(),
        compression_ratio=1.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Compressor class
# ══════════════════════════════════════════════════════════════════════════════

class Compressor:
    """
    BART summarise + BART encoder embeddings → CompressedUnit.
    """

    def __init__(self, device: str | None = None, head_path: str | None = None):
        if device is None:
            device = _default_embed_device()
        self.device = device
        self._head_path = head_path
        self._warmed = False

    def _warm(self) -> None:
        if not self._warmed:
            _get_bart_seq2seq(self.device)
            self._warmed = True

    def _embed(self, text: str) -> torch.Tensor:
        self._warm()
        return _bart_embed_texts([text], device=self.device)[0]

    def _embed_batch(self, texts: list[str]) -> list[torch.Tensor]:
        self._warm()
        return _bart_embed_texts(texts, device=self.device)

    def compress(
        self,
        dialogue_history: str,
        model_response: str,
        max_sentences: int = 8,
        lambda_mmr: float = 0.5,
    ) -> CompressedUnit:
        combined = f"{dialogue_history.strip()}\n{model_response.strip()}".strip()
        raw_sents = _split_sentences(combined) or [combined]
        orig_words = max(1, len(combined.split()))

        raw_vecs = self._embed_batch(raw_sents)
        emb = _stacked_embeddings_matrix(raw_vecs)
        query = _normalize_rows(emb.mean(axis=0, keepdims=True))[0]

        sel_order = _mmr_select(emb, query, max_sentences, lambda_mmr)
        selected_sents = [raw_sents[i] for i in sorted(sel_order)]

        bart_input = " ".join(selected_sents)
        if len(bart_input.split()) > 40:
            self._warm()
            paragraph = _bart_summarize(bart_input, max_length=150, min_length=20, device=self.device)
        else:
            paragraph = bart_input

        para_sents = _split_sentences(paragraph) or [paragraph]
        comp_words = max(1, len(paragraph.split()))

        para_vec = self._embed(paragraph)
        sent_vecs = self._embed_batch(para_sents)
        full_vec = self._embed(combined)

        return CompressedUnit(
            paragraph=paragraph,
            sentences=para_sents,
            paragraph_vector=para_vec,
            sentence_vectors=sent_vecs,
            full_vector=full_vec,
            compression_ratio=orig_words / comp_words,
        )
