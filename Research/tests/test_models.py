"""
DeepShield-X — Model Tests

Verify all models create successfully and produce correct output shapes.
"""

import pytest
import torch
from src.models.factory import create_model, list_available_models, get_model_info


class TestModelFactory:
    """Test the model factory and all registered models."""

    def test_list_available_models(self):
        models = list_available_models()
        assert len(models) >= 6
        assert "custom_cnn" in models
        assert "resnet50" in models

    @pytest.mark.parametrize("model_name", [
        "custom_cnn",
        "resnet50",
        "efficientnet",
        "mobilenet",
    ])
    def test_model_creation(self, model_name, dummy_image):
        """Test that each model creates and produces correct output shape."""
        model = create_model(model_name, pretrained=False, num_classes=1)
        model.eval()

        with torch.no_grad():
            output = model(dummy_image)

        assert output.shape == (1, 1), f"{model_name}: expected (1,1), got {output.shape}"

    def test_custom_cnn_output_shape(self, dummy_batch):
        """Test CustomCNN with a batch."""
        model = create_model("custom_cnn", pretrained=False)
        model.eval()

        with torch.no_grad():
            output = model(dummy_batch)

        assert output.shape == (4, 1)

    def test_model_info(self):
        """Test get_model_info returns expected keys."""
        model = create_model("custom_cnn", pretrained=False)
        info = get_model_info(model, "custom_cnn")

        assert "total_params" in info
        assert "trainable_params" in info
        assert "model_size_mb" in info
        assert info["total_params"] > 0

    def test_invalid_model_raises(self):
        """Test that an unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_finetune_strategy_frozen(self):
        """Test frozen fine-tuning strategy freezes backbone."""
        model = create_model("resnet50", pretrained=False, finetune_strategy="frozen")

        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_count > 0, "Frozen strategy should freeze some parameters"

    def test_gradcam_target_layer(self):
        """Test that models provide a Grad-CAM target layer."""
        for model_name in ["custom_cnn", "resnet50"]:
            model = create_model(model_name, pretrained=False)
            if hasattr(model, "get_gradcam_target_layer"):
                layer = model.get_gradcam_target_layer()
                assert layer is not None
