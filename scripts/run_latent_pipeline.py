#!/usr/bin/env python3
"""
(prompt, bad response) pairs → BART summary of hallu branch → GTR corrector (768-D) →
vec2text decode via ``e2t.embedding_to_text``.

Requires a v4 checkpoint from ``cohesive/training/train.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cohesive.models.hallucination_latent import HalluCorrectorPipeline, TurnResult  # noqa: E402


def _pick_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tensor_to_list(x: torch.Tensor) -> list[float]:
    return x.detach().cpu().float().numpy().tolist()


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _turn_result_to_jsonable(tr: TurnResult) -> dict[str, Any]:
    return {
        "hallucination_risk": tr.hallucination_risk,
        "summary_text": tr.summary_text,
        "anti_hallucination_vector": _tensor_to_list(tr.anti_hallucination_vector),
    }


def run_pipeline(
    pairs: list[tuple[str, str]],
    *,
    accumulate_history: bool = False,
    checkpoint_path: str | None = None,
    device: str | None = None,
    vec2text_steps: int = 20,
) -> list[dict[str, Any]]:
    dev = _pick_device(device)
    ckpt = checkpoint_path or os.environ.get("HALLUCINATION_LATENT_CKPT")
    if ckpt is None:
        ckpt = str(_REPO_ROOT / "checkpoints" / "best.pt")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(
            f"v4 checkpoint required: {ckpt} (train with python cohesive/training/train.py)"
        )
    hls = HalluCorrectorPipeline.load(checkpoint_path=ckpt, device=dev)
    hls.reset_conversation()

    from e2t import embedding_to_text

    messages: list[dict[str, str]] = []
    out: list[dict[str, Any]] = []

    for i, (prompt, response) in enumerate(pairs):
        if not accumulate_history:
            messages = []
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})

        dialogue = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        bad_response = response

        turn = hls.process_turn(dialogue, bad_response)

        try:
            vec = turn.anti_hallucination_vector
            if vec.dim() == 1:
                vec = vec.unsqueeze(0)
            decoded = embedding_to_text(vec.to(dev), num_steps=vec2text_steps)
            decode_error = None
        except Exception as exc:  # noqa: BLE001
            decoded = None
            decode_error = str(exc)

        row: dict[str, Any] = {
            "index": i,
            "prompt": prompt,
            "response": response,
            "turn": _turn_result_to_jsonable(turn),
            "vec2text": decoded,
        }
        if decode_error is not None:
            row["decode_error"] = decode_error
        out.append(row)

    return out


def _parse_pairs_json(raw: str) -> list[tuple[str, str]]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list")
    pairs: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict):
            p = item.get("prompt") or item.get("user") or item.get("q")
            r = item.get("response") or item.get("assistant") or item.get("a")
            if p is None or r is None:
                raise ValueError(f"Missing prompt/response keys in {item!r}")
            pairs.append((str(p), str(r)))
        else:
            raise ValueError(f"Unsupported pair entry: {item!r}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 hallu corrector + vec2text")
    parser.add_argument("--pairs-json", type=str, default=None)
    parser.add_argument("--pairs-file", type=str, default=None)
    parser.add_argument("--accumulate-history", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vec2text-steps", type=int, default=20)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of human summary.",
    )
    args = parser.parse_args()

    default_demo = json.dumps(
        [
            {"prompt": "What is 2+2?", "response": "The moon is made of cheese."},
        ]
    )
    if args.pairs_file:
        raw = Path(args.pairs_file).read_text(encoding="utf-8")
    elif args.pairs_json:
        raw = args.pairs_json
    else:
        raw = default_demo

    pairs = _parse_pairs_json(raw)
    results = run_pipeline(
        pairs,
        accumulate_history=args.accumulate_history,
        checkpoint_path=args.checkpoint,
        device=args.device,
        vec2text_steps=args.vec2text_steps,
    )

    if args.json:
        print(json.dumps(_json_ready(results), indent=2))
    else:
        for row in results:
            print("=" * 72)
            print(f"Turn {row['index']}")
            print("Prompt:", row["prompt"].strip())
            print("Assistant (hallu branch):", row["response"].strip())
            print("BART summary of hallu branch:", row["turn"].get("summary_text", ""))
            print("vec2text(corrector vector):", row.get("vec2text"))
            if row.get("decode_error"):
                print("decode_error:", row["decode_error"])
            print()


if __name__ == "__main__":
    main()
