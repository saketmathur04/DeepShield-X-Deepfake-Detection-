"""
DeepShield-X — Inference Tests

Verify the Predictor class and end-to-end inference.
"""

import numpy as np
import pytest
import torch

from src.models.factory import create_model
from src.inference.predictor import Predictor


class TestPredictor:
    """Test the inference predictor."""

    def test_predictor_creation(self):
        """Test predictor creation without checkpoint."""
        predictor = Predictor(model_name="custom_cnn")
        assert predictor.model is not None

    def test_predict_with_numpy_array(self):
        """Test prediction with a numpy array input."""
        predictor = Predictor(model_name="custom_cnn")
        dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = predictor.predict(dummy_image)

        assert "label" in result
        assert "probability" in result
        assert "confidence" in result
        assert result["label"] in ("REAL", "FAKE")
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["confidence"] <= 1

    def test_predict_returns_valid_json(self):
        """Test that predict returns JSON-serializable output."""
        import json

        predictor = Predictor(model_name="custom_cnn")
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = predictor.predict(dummy_image)
        json_str = json.dumps(result)
        assert json_str is not None

    def test_predict_batch(self):
        """Test batch prediction."""
        predictor = Predictor(model_name="custom_cnn")
        images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]

        results = predictor.predict_batch(images)
        assert len(results) == 3
        for r in results:
            assert "label" in r

    def test_predictor_with_model_instance(self):
        """Test predictor with a pre-loaded model."""
        model = create_model("custom_cnn", pretrained=False)
        predictor = Predictor(model=model)

        dummy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = predictor.predict(dummy)
        assert result["label"] in ("REAL", "FAKE")
