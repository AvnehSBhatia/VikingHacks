"""Load and sanity-check `bge_to_t5_bridge.joblib` without pulling HF models."""

import sys
import unittest
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = PROJECT_ROOT / "bge_to_t5_bridge.joblib"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestBridgeJoblib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BRIDGE_PATH.is_file():
            raise unittest.SkipTest(f"Missing bridge file: {BRIDGE_PATH}")

    def test_loads(self):
        mapper = joblib.load(BRIDGE_PATH)
        self.assertTrue(hasattr(mapper, "predict"))
        self.assertTrue(hasattr(mapper, "coef_"))

    def test_shapes_bge_to_t5(self):
        """BGE-large and gtr-t5-large are both 1024-dim; Ridge multi-output coef is (1024, 1024)."""
        mapper = joblib.load(BRIDGE_PATH)
        self.assertEqual(mapper.coef_.shape, (1024, 1024))
        self.assertEqual(mapper.intercept_.shape, (1024,))

    def test_predict_finite(self):
        mapper = joblib.load(BRIDGE_PATH)
        x = np.random.default_rng(0).standard_normal((3, 1024)).astype(np.float64)
        y = mapper.predict(x)
        self.assertEqual(y.shape, (3, 1024))
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
