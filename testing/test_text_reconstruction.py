"""
End-to-end text reconstruction: BGE encode → Ridge bridge → T5 decode.

Run from repo root (or anywhere; we chdir to project root):

  cd /path/to/VikingHacks && .venv/bin/python -m unittest testing.test_text_reconstruction -v

Skip loading HF models (~minutes) with:

  SKIP_TEXT_RECONSTRUCTION=1 .venv/bin/python -m unittest testing.test_text_reconstruction
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train" / "train.py"
BRIDGE_PATH = PROJECT_ROOT / "bge_to_t5_bridge.joblib"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _has_datasets() -> bool:
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False
    return True


def _load_train_module():
    """Load train/train.py as a module (loads BGE + T5 + bridge once)."""
    spec = importlib.util.spec_from_file_location("viking_train", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TRAIN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _print_pair(original: str, recovered: str, *, cos: float | None = None) -> None:
    print(f"\n[Original]:  {original}", flush=True)
    print(f"[Recovered]: {recovered}", flush=True)
    if cos is not None:
        print(f"[BGE cos]:   {cos:.4f}", flush=True)


@unittest.skipIf(os.environ.get("SKIP_TEXT_RECONSTRUCTION") == "1", "SKIP_TEXT_RECONSTRUCTION=1")
@unittest.skipUnless(_has_datasets(), "datasets package not installed (pip install datasets)")
@unittest.skipUnless(TRAIN_SCRIPT.is_file(), f"missing {TRAIN_SCRIPT}")
@unittest.skipUnless(BRIDGE_PATH.is_file(), f"missing bridge {BRIDGE_PATH}")
class TestTextReconstruction(unittest.TestCase):
    _train = None

    @classmethod
    def setUpClass(cls):
        os.chdir(PROJECT_ROOT)
        # Bridge path in train/train.py is relative to cwd
        cls._train = _load_train_module()

    def test_reconstruct_non_empty(self):
        """Decoded string should not be blank (regression for empty T5 output)."""
        t = self._train
        phrase = "The artificial intelligence system decoded the hidden message successfully."
        vec = t.bge_model.encode([phrase])
        out = t.invert_embedding(vec)
        _print_pair(phrase, out)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out.strip()), 0, msg=f"empty reconstruction for: {phrase!r}")

    def test_reconstruct_multiple_phrases(self):
        t = self._train
        phrases = [
            "Quantum computing uses qubits.",
            "The board approved the annual budget.",
            "Water boils at one hundred degrees Celsius.",
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                vec = t.bge_model.encode([phrase])
                out = t.invert_embedding(vec)
                _print_pair(phrase, out)
                self.assertGreater(
                    len(out.strip()),
                    0,
                    msg=f"empty output for input: {phrase!r}",
                )

    def test_embedding_similarity_sanity(self):
        """
        Recovered text should be somewhat aligned with the source in BGE space
        (not a strict equality test — bridge + T5 are lossy).
        """
        t = self._train
        phrase = "Machine learning models require data and compute."
        src = t.bge_model.encode([phrase])
        text = t.invert_embedding(src)
        tgt = t.bge_model.encode([text])
        a = src.reshape(-1)
        b = tgt.reshape(-1)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        cos = float(np.dot(a, b) / denom)
        _print_pair(phrase, text, cos=cos)
        self.assertGreater(cos, 0.15, msg=f"cos={cos:.3f} recovered={text!r}")


if __name__ == "__main__":
    unittest.main()
