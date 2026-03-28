#!/usr/bin/env python3
"""
(prompt, response) pairs → BART compress → ``HallucinationLatentSpace`` → BART decode.

Requires ``--hallu-checkpoint`` or ``VIKING_HALLU_CHECKPOINT`` (unless ``--dry-run``).

Run::

  python scripts/pipeline_turns.py --json-file pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


Pair = Tuple[str, str]


def _pick_device(preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TurnRecord:
    turn_index: int
    compressed_text: str
    reconstructed_text: str
    tier: str
    extra: dict

    def to_json_dict(self) -> dict:
        return asdict(self)


def iter_pairs_from_args(
    pairs_json: Optional[str],
    json_file: Optional[str],
) -> List[Pair]:
    if json_file:
        path = Path(json_file)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    elif pairs_json:
        data = json.loads(pairs_json)
    else:
        raise ValueError("Provide --pairs-json or --json-file")

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON array")

    out: List[Pair] = []
    for i, item in enumerate(data):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict) and "prompt" in item and "response" in item:
            out.append((str(item["prompt"]), str(item["response"])))
        else:
            raise ValueError(
                f"Invalid pair at index {i}: expected [prompt, response] or objects with prompt/response keys"
            )
    return out


def build_messages_up_to_turn(pairs: Sequence[Pair], turn_index: int) -> list[dict]:
    messages: list[dict] = []
    for i in range(turn_index + 1):
        p, r = pairs[i]
        messages.append({"role": "user", "content": p})
        messages.append({"role": "assistant", "content": r})
    return messages


def run_pipeline(
    pairs: Sequence[Pair],
    *,
    max_tokens: int = 800,
    lambda_mmr: float = 0.5,
    hallu_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    dry_run: bool = False,
) -> list[TurnRecord]:
    if dry_run:
        records: list[TurnRecord] = []
        for t in range(len(pairs)):
            msgs = build_messages_up_to_turn(pairs, t)
            records.append(
                TurnRecord(
                    turn_index=t,
                    compressed_text="",
                    reconstructed_text="",
                    tier="dry_run",
                    extra={"messages_preview": msgs[:2]},
                )
            )
        return records

    if not hallu_checkpoint or not Path(hallu_checkpoint).is_file():
        raise FileNotFoundError(
            "Hallucination checkpoint is required: pass --hallu-checkpoint or set VIKING_HALLU_CHECKPOINT "
            f"(got {hallu_checkpoint!r})"
        )

    from cohesive import try_load_hallucination_latent_space
    from cohesive.models.compressor import bart_decode_from_vector, compressed_unit_from_paragraph

    from compressor import compress

    dev = _pick_device(device)
    hls = try_load_hallucination_latent_space(hallu_checkpoint, device=dev)
    if hasattr(hls, "reset_conversation"):
        hls.reset_conversation()

    records = []
    for t in range(len(pairs)):
        messages = build_messages_up_to_turn(pairs, t)
        compressed_text = compress(messages, max_tokens=max_tokens, lambda_mmr=lambda_mmr, device=dev)

        extra: dict = {}
        if not compressed_text.strip():
            records.append(
                TurnRecord(
                    turn_index=t,
                    compressed_text=compressed_text,
                    reconstructed_text="",
                    tier="hallu_bart",
                    extra={"warning": "empty_compression"},
                )
            )
            continue

        unit = compressed_unit_from_paragraph(compressed_text, device=dev)
        result = hls.process_turn(unit)
        anti = result.anti_hallucination_vector
        if hasattr(anti, "detach"):
            anti = anti.detach().cpu()
        recon = bart_decode_from_vector(anti, device=dev)

        if hasattr(hls, "update_stretch"):
            sent_latent = [hls.encode(sv) for sv in unit.sentence_vectors]
            hls.update_stretch(sent_latent, unit.sentences, is_hallucination=False)

        extra = {
            "sentence_scores": getattr(result, "sentence_scores", None),
            "log_det": float(getattr(result, "log_det", 0.0)) if hasattr(result, "log_det") else None,
        }
        records.append(
            TurnRecord(
                turn_index=t,
                compressed_text=compressed_text,
                reconstructed_text=recon,
                tier="hallu_bart",
                extra=extra,
            )
        )

    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="BART compress + HallucinationLatentSpace + BART decode (checkpoint required)."
    )
    p.add_argument("--json-file", type=str, default=None)
    p.add_argument("--pairs-json", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--lambda-mmr", type=float, default=0.5)
    p.add_argument("--hallu-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.json_file and not args.pairs_json:
        p.error("Provide --json-file and/or --pairs-json (JSON array of pairs).")

    default_ckpt = os.environ.get("VIKING_HALLU_CHECKPOINT")
    hallu_ckpt = args.hallu_checkpoint or default_ckpt
    if not args.dry_run and (not hallu_ckpt or not Path(hallu_ckpt).is_file()):
        p.error(
            "Provide a valid --hallu-checkpoint path or set VIKING_HALLU_CHECKPOINT "
            f"(file not found: {hallu_ckpt!r})"
        )

    pairs = iter_pairs_from_args(args.pairs_json, args.json_file)
    records = run_pipeline(
        pairs,
        max_tokens=args.max_tokens,
        lambda_mmr=args.lambda_mmr,
        hallu_checkpoint=hallu_ckpt if hallu_ckpt else None,
        device=args.device,
        dry_run=args.dry_run,
    )

    out = [r.to_json_dict() for r in records]
    text = json.dumps(out, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
