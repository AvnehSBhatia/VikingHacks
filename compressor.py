"""Compress raw chat history into one fluent paragraph with a 3-stage pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Import the shared interface module so this compressor module stays aligned
# with the project's compression interface layer.
from compressor_interface import CompressorInterface  # noqa: F401

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def _resolve_hf_token() -> str | None:
    """Load .env and return a Hugging Face token if available."""
    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
        load_dotenv()
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )


_HF_TOKEN = _resolve_hf_token()


def _init_embed_model() -> SentenceTransformer:
    """Initialize sentence-transformer with token support across versions."""
    if not _HF_TOKEN:
        return SentenceTransformer("all-mpnet-base-v2")

    try:
        return SentenceTransformer("all-mpnet-base-v2", token=_HF_TOKEN)
    except TypeError:
        return SentenceTransformer("all-mpnet-base-v2", use_auth_token=_HF_TOKEN)


def _build_pipeline(task: str | None):
    """Create transformers pipeline with token compatibility across versions."""
    if _HF_TOKEN:
        try:
            if task is None:
                return pipeline(model="facebook/bart-large-cnn", token=_HF_TOKEN)
            return pipeline(task, model="facebook/bart-large-cnn", token=_HF_TOKEN)
        except TypeError:
            if task is None:
                return pipeline(model="facebook/bart-large-cnn", use_auth_token=_HF_TOKEN)
            return pipeline(task, model="facebook/bart-large-cnn", use_auth_token=_HF_TOKEN)

    if task is None:
        return pipeline(model="facebook/bart-large-cnn")
    return pipeline(task, model="facebook/bart-large-cnn")


# Load models once at module import time.
_EMBED_MODEL = _init_embed_model()


def _init_summarizer():
    """Initialize a BART summarizer pipeline across transformers versions."""
    # Different transformers builds expose different task names.
    candidate_tasks = ["summarization", "text2text-generation", "any-to-any", "text-generation"]
    for task in candidate_tasks:
        try:
            summarizer = _build_pipeline(task)
            return summarizer
        except Exception:
            continue

    # Final attempt: allow transformers to infer task from model config.
    try:
        return _build_pipeline(None)
    except Exception as exc:
        raise RuntimeError(
            "Could not create a transformers pipeline for facebook/bart-large-cnn in this environment."
        ) from exc


_SUMMARIZER = _init_summarizer()


def compress(messages: list[dict], target_count: int, lambda_mmr: float = 0.5) -> str:
    """Compress conversation history via sliding-window scoring, MMR, and BART.

    Inputs:
        messages: list[dict] where each item includes role (str) and content (str).
        target_count: number of messages MMR should select before summarization.
        lambda_mmr: relevance/diversity tradeoff for MMR (default 0.5).
    Output:
        compressed_text: single fluent paragraph summary string.

    Stages:
        1) Sliding Window Scoring: local relevance from avg cosine similarity to neighbors.
        2) MMR Selection: select diverse-yet-relevant messages using numpy implementation.
        3) BART Summarization: summarize ordered selected messages with facebook/bart-large-cnn.
    """
    if not messages:
        return ""

    if target_count <= 0:
        return ""

    if not 0.0 <= lambda_mmr <= 1.0:
        raise ValueError("lambda_mmr must be between 0.0 and 1.0")

    # Stage 1: embed each message and score with local sliding-window similarity.
    contents = [str(msg.get("content", "")) for msg in messages]
    embeddings = _EMBED_MODEL.encode(contents, convert_to_numpy=True)

    window_scores: list[float] = []
    for i in range(len(messages)):
        similarities: list[float] = []
        left = max(0, i - 2)
        right = min(len(messages), i + 3)
        for j in range(left, right):
            if j == i:
                continue
            a = embeddings[i]
            b = embeddings[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            sim = float(np.dot(a, b) / denom) if denom != 0.0 else 0.0
            similarities.append(sim)
        window_scores.append(float(np.mean(similarities)) if similarities else 0.0)

    # Stage 2: run MMR selection from scratch (numpy-only math, no MMR library).
    k = min(target_count, len(messages))
    remaining = list(range(len(messages)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -float("inf")

        for candidate in remaining:
            relevance_score = window_scores[candidate]
            if not selected:
                max_similarity = 0.0
            else:
                candidate_vec = embeddings[candidate]
                max_similarity = -float("inf")
                for chosen in selected:
                    chosen_vec = embeddings[chosen]
                    denom = float(np.linalg.norm(candidate_vec) * np.linalg.norm(chosen_vec))
                    sim = float(np.dot(candidate_vec, chosen_vec) / denom) if denom != 0.0 else 0.0
                    if sim > max_similarity:
                        max_similarity = sim
                if max_similarity == -float("inf"):
                    max_similarity = 0.0

            mmr_score = (lambda_mmr * relevance_score) - ((1.0 - lambda_mmr) * max_similarity)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = candidate

        selected.append(best_idx)
        remaining.remove(best_idx)

    # Restore original conversational order for final summarization input.
    selected.sort()

    # Stage 3: build role-prefixed text and summarize with BART.
    ordered_chunks = []
    for idx in selected:
        role = str(messages[idx].get("role", "unknown")).strip().capitalize()
        content = str(messages[idx].get("content", "")).strip()
        ordered_chunks.append(f"{role}: {content}")
    bart_input = " ".join(ordered_chunks)

    summary = _SUMMARIZER(bart_input)
    first = summary[0] if isinstance(summary, list) else summary
    if isinstance(first, dict):
        if "summary_text" in first:
            return str(first["summary_text"]).strip()
        if "generated_text" in first:
            return str(first["generated_text"]).strip()
        if "text" in first:
            return str(first["text"]).strip()
        # Last-resort fallback for unexpected pipeline output shapes.
        return str(first).strip()
    return str(first).strip()
