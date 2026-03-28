"""
Smoke tests for cohesive v4: GTR 768-D space, stretch, HalluCorrectorModule.

Does not require checkpoints. Full pipeline + HF models are optional (see RUN_COHESIVE_GTR=1).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cohesive.constants import CKPT_VERSION_V4, EMBED_DIM
from cohesive.models.hallu_corrector_module import HalluCorrectorModule
from cohesive.models.hallucination_latent import (
    BGE_DIM,
    CKPT_VERSION,
    LATENT_DIM,
    StretchMatrix,
)


class TestCohesiveV4(unittest.TestCase):
    def test_dims_and_version(self):
        self.assertEqual(EMBED_DIM, 768)
        self.assertEqual(BGE_DIM, LATENT_DIM)
        self.assertEqual(CKPT_VERSION, CKPT_VERSION_V4)

    def test_stretch_apply(self):
        st = StretchMatrix(dim=EMBED_DIM)
        z = F.normalize(torch.randn(EMBED_DIM), dim=0)
        out = st.apply(z)
        self.assertEqual(out.shape, (EMBED_DIM,))

    def test_corrector_shapes(self):
        cor = HalluCorrectorModule(EMBED_DIM)
        z = F.normalize(torch.randn(2, EMBED_DIM), dim=-1)
        y = cor(z)
        self.assertEqual(y.shape, (2, EMBED_DIM))

    def test_corrector_state_roundtrip(self):
        m = HalluCorrectorModule(EMBED_DIM)
        m.S.data = m.S.data + 0.01 * torch.randn_like(m.S)
        d = m.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(d, path)
            m2 = HalluCorrectorModule(EMBED_DIM)
            try:
                loaded = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                loaded = torch.load(path, map_location="cpu")
            m2.load_state_dict(loaded)
            self.assertTrue(torch.allclose(m.S, m2.S))
        finally:
            Path(path).unlink(missing_ok=True)


@unittest.skipUnless(os.environ.get("RUN_COHESIVE_GTR") == "1", "set RUN_COHESIVE_GTR=1 to load GTR + pipeline")
class TestCohesiveGTRIntegration(unittest.TestCase):
    def test_pipeline_construct_and_forward(self):
        from cohesive.models.hallucination_latent import HalluCorrectorPipeline
        from cohesive.models.sentence_encoder import GTRSentenceEncoder

        enc = GTRSentenceEncoder(device="cpu")
        cor = HalluCorrectorModule(EMBED_DIM)
        pipe = HalluCorrectorPipeline(enc, cor, device="cpu")
        z = F.normalize(torch.randn(1, EMBED_DIM), dim=-1)
        out = pipe.corrector(z)
        self.assertEqual(out.shape, (1, EMBED_DIM))


if __name__ == "__main__":
    unittest.main()
