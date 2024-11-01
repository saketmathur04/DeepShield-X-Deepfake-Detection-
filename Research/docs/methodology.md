# DeepShield-X — Methodology

## 1. Problem Definition

DeepFake detection is formulated as a **binary image classification** task:
- **Input**: Facial image (cropped, resized to 224×224)
- **Output**: REAL (0) or FAKE (1) with prediction probability
- **Positive class**: FAKE
- **Negative class**: REAL

## 2. Dataset

### Primary Dataset: FaceForensics++
- Source: Rössler et al., ICCV 2019
- Contains real YouTube face videos and manipulated versions
- Manipulation methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- Compression levels: c0 (raw), c23 (light), c40 (heavy)

### Secondary Dataset: Celeb-DF
- Source: Li et al., CVPR 2020
- Used exclusively for cross-dataset generalization testing
- Never mixed into training data

## 3. Preprocessing Pipeline

1. **Frame extraction**: Sample every Nth frame from source videos
2. **Face detection**: OpenCV Haar cascade classifier
3. **Face cropping**: With 30% margin around detected face bounding box
4. **Resizing**: To target resolution (default: 224×224)
5. **Normalization**: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## 4. Data Splitting

**Source-aware splitting** to prevent data leakage:
- Split at the VIDEO level (not frame level)
- All frames from one source video → same split
- Default ratios: Train 70% / Val 15% / Test 15%
- Automated leakage detection: raises error if any source appears in multiple splits
- Stratified by label to maintain class balance

## 5. Data Augmentation

Using Albumentations library:
- Horizontal flip (p=0.5)
- Rotation ±15° (p=0.5)
- Brightness/Contrast ±20% (p=0.5)
- Gaussian blur, kernel≤3 (p=0.3)
- JPEG compression, quality 60-100 (p=0.3)
- Hue/Saturation jitter (p=0.3)

Augmentation is **configurable** — can be enabled/disabled for controlled comparison.

## 6. Model Architectures

All models share a common interface through the model factory pattern.

| Architecture | Pretrained | Key Feature |
|-------------|-----------|-------------|
| Custom CNN | No | 3-block baseline (Conv→BN→ReLU→Pool) |
| ResNet-50 | ImageNet | Residual connections |
| EfficientNet-B0 | ImageNet | Compound scaling |
| MobileNetV3-Small | ImageNet | Depthwise separable convolutions |
| Xception | ImageNet | Depthwise separable + cross-channel correlation |
| ViT-Small/16 | ImageNet | Self-attention on image patches |

## 7. Training Configuration

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine annealing
- **Loss**: Binary cross-entropy with logits
- **Mixed precision**: Enabled on CUDA
- **Early stopping**: Patience=5 epochs, monitoring validation loss
- **Gradient clipping**: max_norm=1.0
- **Batch size**: 32 (configurable)
- **Epochs**: 20 (configurable)

## 8. Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Specificity, Sensitivity
- Confusion matrix
- Inference latency

## 9. Explainability

**Grad-CAM** (Gradient-weighted Class Activation Mapping):
- Generates heatmaps showing which image regions influenced the prediction
- Applied to the last convolutional layer (CNN) or last transformer block (ViT)
- Output: original image + heatmap overlay

## 10. Reproducibility

- Fixed random seed (42) across all libraries
- Environment info logged (Python, PyTorch, CUDA versions)
- Full configuration saved with each experiment
- Deterministic CUDA operations enabled
