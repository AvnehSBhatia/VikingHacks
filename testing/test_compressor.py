import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDING_MAP = {}
SUMMARY_INPUTS = []


def fake_bart_embed_texts(texts: list[str], device: str | None = None):
    """Deterministic 2-D or 1024-D vectors for MMR tests (no HF BART load)."""
    out = []
    for t in texts:
        if t in EMBEDDING_MAP:
            v = np.array(EMBEDDING_MAP[t], dtype=float)
        else:
            h = hash(t) % (2**31)
            rng = np.random.default_rng(h)
            v = rng.standard_normal(1024).astype(float)
        out.append(torch.from_numpy(v).float())
    return out


class FakeSummarizer:
    def __call__(self, text, **kwargs):
        SUMMARY_INPUTS.append(text)
        return [{"summary_text": f"SUMMARY::{text}"}]


def fake_pipeline(*args, **kwargs):
    return FakeSummarizer()


class FakeTokenizer:
    """Minimal tokenizer for BART length logic in tests."""

    model_max_length = 1024

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(
        self,
        text,
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
    ):
        if not text:
            return []
        n = min(len(text.split()), max_length or 1024)
        return list(range(max(1, n)))


def import_compressor_with_fakes():
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline
    fake_transformers.AutoTokenizer = FakeTokenizer

    sys.modules["transformers"] = fake_transformers
    for name in (
        "compressor",
        "cohesive.models.compressor",
    ):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    import cohesive.models.compressor as cm

    cm._bart_embed_texts = fake_bart_embed_texts
    return importlib.import_module("compressor")


class CompressorTests(unittest.TestCase):
    def setUp(self):
        SUMMARY_INPUTS.clear()
        EMBEDDING_MAP.clear()
        self.compressor = import_compressor_with_fakes()

    def test_returns_empty_string_for_empty_messages(self):
        result = self.compressor.compress([], target_count=2)
        self.assertEqual(result, "")
        self.assertEqual(SUMMARY_INPUTS, [])

    def test_raises_for_invalid_lambda(self):
        with self.assertRaises(ValueError):
            self.compressor.compress([{"role": "user", "content": "hello"}], 1, lambda_mmr=1.5)

    def test_mmr_selection_restores_original_order_before_bart(self):
        messages = [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "m1"},
            {"role": "assistant", "content": "m2"},
        ]
        EMBEDDING_MAP.update(
            {
                "m0": [1.0, 0.0],
                "m1": [0.0, 1.0],
                "m2": [1.0, 1.0],
            }
        )

        result = self.compressor.compress(messages, target_count=2, lambda_mmr=1.0)

        self.assertEqual(len(SUMMARY_INPUTS), 1)
        self.assertEqual(SUMMARY_INPUTS[0], "User: m0 Assistant: m2")
        self.assertEqual(result, "SUMMARY::User: m0 Assistant: m2")


def print_examples() -> None:
    """Print sample compressor inputs and outputs (uses fakes; no HF download)."""
    print("=== compress(messages, target_count) ===\n")
    SUMMARY_INPUTS.clear()
    EMBEDDING_MAP.clear()
    mod = import_compressor_with_fakes()

    print("Empty history:")
    print(f"  compress([], target_count=2) -> {mod.compress([], target_count=2)!r}\n")

    messages = [
        {"role": "user", "content": "m0"},
        {"role": "assistant", "content": "m1"},
        {"role": "assistant", "content": "m2"},
    ]
    EMBEDDING_MAP.update(
        {
            "m0": [1.0, 0.0],
            "m1": [0.0, 1.0],
            "m2": [1.0, 1.0],
        }
    )
    out = mod.compress(messages, target_count=2, lambda_mmr=1.0)
    print("After MMR + BART summarizer (fake):")
    print(f"  BART input: {SUMMARY_INPUTS[0]!r}")
    print(f"  compress(...) -> {out!r}\n")

    print("=== Compressor.compress(dialogue, response) -> CompressedUnit ===\n")
    print("  (loads real BART — run locally if needed)\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        print_examples()
    else:
        unittest.main()
