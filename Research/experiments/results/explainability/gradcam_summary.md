# DeepShield-X — Explainability via Grad-CAM Analysis

Visualizing spatial saliency of ResNet50 Regularized (`layer4` feature activations).

| Sample | Path | Predicted Class | Confidence | Salient Regions |
|---|---|---|---|---|
| #1 | `data\faces\test\fake\Deepfakes\040_997\frame_0000.jpg` | FAKE | 1.0000 | Facial Boundary / Perioral / Nasal |
| #2 | `data\faces\test\fake\Deepfakes\040_997\frame_0054.jpg` | FAKE | 1.0000 | Facial Boundary / Perioral / Nasal |
| #3 | `data\faces\test\fake\Deepfakes\040_997\frame_0108.jpg` | FAKE | 1.0000 | Facial Boundary / Perioral / Nasal |
| #4 | `data\faces\test\fake\Deepfakes\040_997\frame_0162.jpg` | FAKE | 0.9995 | Facial Boundary / Perioral / Nasal |
| #5 | `data\faces\test\fake\Deepfakes\040_997\frame_0216.jpg` | FAKE | 1.0000 | Facial Boundary / Perioral / Nasal |
| #6 | `data\faces\test\fake\Deepfakes\040_997\frame_0270.jpg` | FAKE | 0.8990 | Facial Boundary / Perioral / Nasal |

### Scientific Assessment:
- **Salient Regions**: Saliency heatmaps prominently focus on facial boundaries (chin, jawline, hairline) and perioral / eye regions where blending artifacts and warping discontinuities typically emerge.
- **Interpretability Caveat**: Saliency maps highlight regions correlated with model activation; they indicate influential receptive field clusters rather than semantic human reasoning.
