import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pls.editflow.runtime_cost import (
    load_runtime_cost_model,
    predict_runtime_seconds,
    runtime_cost_scale,
)


class RuntimeCostTest(unittest.TestCase):
    def _model(self):
        return {
            "schema": "PLS_ESMFold_monotone_runtime_cost_model_v1",
            "length_knots": [50.0, 100.0, 200.0],
            "seconds_knots": [1.0, 2.0, 8.0],
            "reference_cost_seconds": 2.0,
            "test_evaluated": False,
        }

    def test_prediction_is_clipped_and_monotone(self):
        predicted = predict_runtime_seconds(self._model(), [25, 50, 75, 200, 300])
        np.testing.assert_allclose(predicted, [1.0, 1.0, 1.5, 8.0, 8.0])
        self.assertTrue(np.all(np.diff(predicted) >= 0))

    def test_positive_gamma_tightens_expensive_anchors(self):
        scale = runtime_cost_scale(self._model(), [50, 100, 200], gamma=1.0)
        np.testing.assert_allclose(scale, [2.0, 1.0, 0.25])

    def test_model_validation_rejects_nonmonotone_cost(self):
        model = self._model()
        model["seconds_knots"] = [1.0, 3.0, 2.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(model))
            with self.assertRaises(ValueError):
                load_runtime_cost_model(path)


if __name__ == "__main__":
    unittest.main()
