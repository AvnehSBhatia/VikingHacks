"""Load ``checkpoints/best.pt`` and sanity-check v4 GTR corrector (no full BART path in unit tests).

**Report** (loads checkpoint, optional ``process_turn`` with real summarizer — slow):

  python testing/test_checkpoint_best.py
  python testing/test_checkpoint_best.py --checkpoint path/to.pt --device cpu

**Unit tests** (lightweight: keys + corrector forward on random z):

  python testing/test_checkpoint_best.py --test
  python -m unittest testing.test_checkpoint_best -v
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cosine_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp(min=1e-8)
    return float((a * b).sum() / denom)


def _tensor_brief(t: torch.Tensor, name: str) -> str:
    t = t.detach().float().reshape(-1)
    return (
        f"  {name}: shape={tuple(t.shape)}  "
        f"L2={float(t.norm()):.6f}  "
        f"min={float(t.min()):.4f}  max={float(t.max()):.4f}"
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    seed: int = 0,
):
    """
    Load v4 checkpoint, run corrector on a random normalised vector (matches embed scale).

    Returns ``(z_in, z_out, metrics_dict)``.
    """
    from cohesive.models.hallucination_latent import BGE_DIM, HalluCorrectorPipeline

    g = torch.Generator().manual_seed(seed)
    pipe = HalluCorrectorPipeline.load(str(checkpoint_path), device=device)
    pipe.reset_conversation()
    z = torch.randn(1, BGE_DIM, generator=g, device=device, dtype=torch.float32)
    z = F.normalize(z, dim=-1)
    with torch.no_grad():
        z_hat = pipe.corrector(z)
    z_in = z.squeeze(0).cpu()
    z_out = z_hat.squeeze(0).cpu()

    metrics = {
        "cosine_in_out": _cosine_1d(z_in, z_out),
        "out_L2": float(z_out.norm()),
        "bge_dim": BGE_DIM,
    }
    return z_in, z_out, metrics


def evaluate_checkpoint_process_turn(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    dialogue: str = "User: What is 2+2?\nAssistant: ",
    bad_response: str = "The answer is five.",
):
    """Full path: BART summary + GTR embed + corrector (downloads / slow)."""
    from cohesive.models.hallucination_latent import BGE_DIM, HalluCorrectorPipeline

    pipe = HalluCorrectorPipeline.load(str(checkpoint_path), device=device)
    pipe.reset_conversation()
    out = pipe.process_turn(dialogue, bad_response)
    v = out.anti_hallucination_vector.reshape(-1).float()
    metrics = {
        "summary_len": len(out.summary_text or ""),
        "out_L2": float(v.norm()),
        "bge_dim": BGE_DIM,
    }
    return out, metrics


def print_checkpoint_report(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    full_turn: bool = False,
    stream=None,
):
    """Print a human-readable report; returns metrics dict (and tensors if not full_turn)."""
    if stream is None:
        stream = sys.stdout

    def p(*args, **kwargs):
        kwargs.setdefault("file", stream)
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")

    p("=" * 72)
    p("CHECKPOINT REPORT (v4 GTR corrector)")
    p("=" * 72)
    p(f"Path:     {checkpoint_path}")
    p(f"Device:   {device}")
    p()
    p("--- checkpoint bundle ---")
    p(f"  state keys: {sorted(raw.keys())}")
    ver = int(raw.get("ckpt_version", 0))
    p(f"  ckpt_version: {ver}")
    if ver < 4 or "corrector" not in raw:
        p()
        p("  This file is not a v4 checkpoint (need ckpt_version >= 4 and 'corrector' key).")
        p("  Train with: python cohesive/training/train.py")
        p("=" * 72)
        return {}
    if "corrector" in raw and "S" in raw["corrector"]:
        S = raw["corrector"]["S"]
        p(f"  corrector.S shape: {tuple(S.shape)}")
    if "stretch_session" in raw and "S" in raw["stretch_session"]:
        Ss = raw["stretch_session"]["S"]
        p(f"  stretch_session.S shape: {tuple(Ss.shape)}")
    p()

    if full_turn:
        out, m = evaluate_checkpoint_process_turn(checkpoint_path, device=device)
        p("--- process_turn (BART + GTR + corrector) ---")
        p(f"  summary_text (first 200 chars): {(out.summary_text or '')[:200]!r}")
        p(_tensor_brief(out.anti_hallucination_vector, "anti_hallucination_vector"))
        p(f"  out_L2: {m['out_L2']:.6f}")
    else:
        z_in, z_out, m = evaluate_checkpoint(checkpoint_path, device=device, seed=seed)
        p(f"--- corrector only (seed={seed}, random z on unit sphere) ---")
        p(_tensor_brief(z_in, "z_in (random normalised)"))
        p(_tensor_brief(z_out, "z_hat (corrector out)"))
        p(f"  cos(z_in, z_hat): {m['cosine_in_out']:+.6f}")
        p(f"  ||z_hat||:        {m['out_L2']:.6f}")
    p("=" * 72)
    return m


def run_checkpoint_report_main() -> None:
    ap = argparse.ArgumentParser(description="Print v4 checkpoint report.")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT,
        help=f"path to .pt (default: {CHECKPOINT})",
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for random z in fast path")
    ap.add_argument(
        "--full-turn",
        action="store_true",
        help="Run BART summarizer + process_turn (slow, needs transformers)",
    )
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        print(f"Missing checkpoint: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    print_checkpoint_report(
        args.checkpoint, device=args.device, seed=args.seed, full_turn=args.full_turn
    )


def _checkpoint_is_v4(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    ver = int(data.get("ckpt_version", 0))
    return ver >= 4 and "corrector" in data and "stretch_session" in data


class TestBestCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.is_file():
            raise unittest.SkipTest(f"Missing checkpoint: {CHECKPOINT}")
        if not _checkpoint_is_v4(CHECKPOINT):
            raise unittest.SkipTest(
                f"{CHECKPOINT} is not a v4 checkpoint (need ckpt_version>=4 with "
                "encoder, corrector, stretch_session). Train with cohesive/training/train.py."
            )

    def test_raw_state_keys(self):
        data = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        for k in ("encoder", "corrector", "stretch_session", "ckpt_version"):
            self.assertIn(k, data, msg=f"missing state key {k!r}")
        self.assertGreaterEqual(int(data["ckpt_version"]), 4)

    def test_load_and_corrector_forward(self):
        from cohesive.models.hallucination_latent import BGE_DIM, HalluCorrectorPipeline

        pipe = HalluCorrectorPipeline.load(str(CHECKPOINT), device="cpu")
        z = F.normalize(torch.randn(1, BGE_DIM), dim=-1)
        with torch.no_grad():
            v = pipe.corrector(z)
        self.assertEqual(v.shape, (1, BGE_DIM))
        self.assertTrue(torch.isfinite(v).all())


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--test"]
        unittest.main()
    else:
        run_checkpoint_report_main()
