"""
DeepShield-X — Streamlit Demo

A simple, professional demo for deepfake detection.
Upload an image → get REAL/FAKE prediction + Grad-CAM explanation.

Usage:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DeepShield-X — DeepFake Detection",
    page_icon="🛡️",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .fake-prediction {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .real-prediction {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .metric-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_predictor():
    """Load the prediction model (cached)."""
    from src.inference.predictor import Predictor
    from src.utils.device import get_device

    device = get_device()

    # Try to find a checkpoint
    ckpt_dir = Path("experiments/checkpoints")
    checkpoint = None

    if ckpt_dir.exists():
        for ckpt in ckpt_dir.rglob("*_best.pth"):
            checkpoint = str(ckpt)
            break

    model_name = st.session_state.get("model_name", "resnet50")

    predictor = Predictor(
        checkpoint_path=checkpoint,
        model_name=model_name,
        device=device,
    )

    return predictor


def main():
    # Header
    st.markdown("<div class='main-header'>", unsafe_allow_html=True)
    st.title("🛡️ DeepShield-X")
    st.markdown("**Robust & Explainable DeepFake Detection**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        model_name = st.selectbox(
            "Model Architecture",
            ["resnet50", "efficientnet", "mobilenet", "custom_cnn", "xception", "vit"],
            index=0,
        )
        st.session_state["model_name"] = model_name

        show_gradcam = st.checkbox("Show Grad-CAM Explanation", value=True)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "DeepShield-X uses deep learning to detect manipulated "
            "facial images (deepfakes). Upload a face image to analyze."
        )

        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown(
            "*This tool is for research purposes only. "
            "Results should not be treated as absolute proof of manipulation.*"
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a face image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload a facial image to analyze for deepfake manipulation",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Analysis Result")

            with st.spinner("Analyzing image..."):
                try:
                    predictor = load_predictor()
                    img_array = np.array(image)

                    if show_gradcam:
                        result = predictor.predict_with_gradcam(img_array)
                    else:
                        result = predictor.predict(img_array)

                    # Display prediction
                    label = result.get("prediction", result.get("label", "UNKNOWN"))
                    confidence = result.get("confidence", 0)
                    probability = result.get("probability", 0)

                    if label == "FAKE":
                        st.error(f"🚨 **Prediction: {label}**")
                        st.metric("Confidence", f"{confidence:.1%}")
                    else:
                        st.success(f"✅ **Prediction: {label}**")
                        st.metric("Confidence", f"{confidence:.1%}")

                    # Progress bar for confidence
                    st.progress(confidence)

                    # Details
                    with st.expander("📊 Detailed Results"):
                        st.json({
                            "prediction": label,
                            "confidence": f"{confidence:.4f}",
                            "fake_probability": f"{probability:.4f}",
                            "model": model_name,
                        })

                    # Grad-CAM
                    if show_gradcam and "overlay" in result and result["overlay"] is not None:
                        st.subheader("🔬 Grad-CAM Explanation")
                        st.image(
                            result["overlay"],
                            caption="Regions influencing the prediction",
                            use_container_width=True,
                        )
                        st.caption(
                            "Warm colors (red/yellow) indicate regions that most "
                            "influenced the model's decision."
                        )

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Make sure you have a trained model or run with default pretrained weights.")


if __name__ == "__main__":
    main()
