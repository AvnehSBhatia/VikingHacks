import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is importable even when running from `testing/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDING_MAP = {}
SUMMARY_INPUTS = []


class FakeSentenceTransformer:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def encode(self, sentences, convert_to_numpy=True):
        vectors = [np.array(EMBEDDING_MAP[s], dtype=float) for s in sentences]
        if convert_to_numpy:
            return np.vstack(vectors)
        return vectors


class FakeSummarizer:
    def __call__(self, text):
        SUMMARY_INPUTS.append(text)
        return [{"summary_text": f"SUMMARY::{text}"}]


def fake_pipeline(*args, **kwargs):
    return FakeSummarizer()


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def import_compressor_with_fakes():
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline
    fake_transformers.AutoTokenizer = FakeAutoTokenizer

    sys.modules["sentence_transformers"] = fake_sentence_transformers
    sys.modules["transformers"] = fake_transformers
    sys.modules.pop("compressor", None)
    return importlib.import_module("compressor")


class CompressorTests(unittest.TestCase):
    def setUp(self):
        SUMMARY_INPUTS.clear()
        EMBEDDING_MAP.clear()
        self.compressor = import_compressor_with_fakes()

    def test_returns_empty_string_for_empty_messages(self):
        result = self.compressor.compress([], max_tokens=20)
        self.assertEqual(result, "")
        self.assertEqual(SUMMARY_INPUTS, [])

    def test_raises_for_invalid_lambda(self):
        with self.assertRaises(ValueError):
            self.compressor.compress([{"role": "user", "content": "hello"}], 20, lambda_mmr=1.5)

    def test_mmr_selection_restores_original_order_before_bart(self):
        messages = [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "m1"},
            {"role": "assistant", "content": "m2"},
        ]
        # Embeddings chosen so index 2 has highest window score, then index 0.
        EMBEDDING_MAP.update(
            {
                "m0": [1.0, 0.0],
                "m1": [0.0, 1.0],
                "m2": [1.0, 1.0],
            }
        )

        result = self.compressor.compress(messages, max_tokens=4, lambda_mmr=1.0)

        self.assertEqual(len(SUMMARY_INPUTS), 1)
        # If order is restored after MMR, selected [2,0] becomes [0,2].
        self.assertEqual(SUMMARY_INPUTS[0], "User: m0 Assistant: m2")
        self.assertEqual(result, "SUMMARY::User: m0 Assistant: m2")

    def test_mmr_stops_when_next_candidate_would_exceed_max_tokens(self):
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

        # One message is 2 tokens ("User: m0"), two messages would be 4 tokens.
        result = self.compressor.compress(messages, max_tokens=2, lambda_mmr=1.0)

        self.assertEqual(SUMMARY_INPUTS[0], "Assistant: m2")
        self.assertEqual(result, "SUMMARY::Assistant: m2")


if __name__ == "__main__":
    unittest.main()
