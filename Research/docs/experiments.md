# DeepShield-X — Experiments Documentation

## Experimental Design

All experiments follow a controlled methodology:
- Same dataset splits across all experiments
- Fixed random seed for reproducibility
- Results saved automatically (JSON, CSV, Markdown, PNG plots)
- Each experiment can be run independently or as part of the master runner

---

## Experiment 1: Model Comparison

**Objective**: Compare all 6 architectures on the same dataset.

**Setup**:
- Models: Custom CNN, ResNet-50, EfficientNet-B0, MobileNetV3, Xception, ViT
- Same training configuration for all models
- Same train/val/test splits

**Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Parameters, Inference Time

**Output**:
- Comparison table (CSV, Markdown)
- Per-metric bar charts
- Multi-metric grouped bar chart

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 2: Data Augmentation Impact

**Objective**: Measure the effect of data augmentation on model performance.

**Setup**:
- Baseline: No augmentation
- Augmented: Realistic augmentations (flip, rotation, brightness, blur, JPEG, color)
- Model: ResNet-50 (or best-performing model)

**Output**:
- Accuracy/F1 comparison
- ROC curve comparison
- Training curves comparison

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 3: Image Resolution Impact

**Objective**: Study how input resolution affects performance and efficiency.

**Setup**:
- Resolutions: 128×128, 224×224, 384×384
- Model: ResNet-50

**Output**:
- Resolution vs Performance plots
- Resolution vs Inference Time plots

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 4: JPEG Compression Robustness

**Objective**: Test robustness to JPEG compression artifacts.

**Setup**:
- Quality levels: 100, 80, 60, 40, 20
- Applied to test set only (dynamically, not permanently)
- Model: Best-performing model

**Output**:
- JPEG Quality vs Accuracy/F1/ROC-AUC curves

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 5: Blur & Noise Robustness

**Objective**: Test robustness to image degradation.

**Setup**:
- Gaussian blur: kernel sizes [1, 3, 5, 7, 9]
- Gaussian noise: σ = [0.01, 0.03, 0.05, 0.1, 0.15]
- Applied dynamically to test set

**Output**:
- Blur severity vs F1 curve
- Noise level vs F1 curve

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 6: Cross-Dataset Generalization (Optional)

**Objective**: Test if models trained on FaceForensics++ generalize to Celeb-DF.

**Setup**:
- Train: FaceForensics++
- Test: Celeb-DF (NO retraining)
- Model: Best-performing model

**Label**: EXTERNAL GENERALIZATION EXPERIMENT

**Output**: Full metrics report, confusion matrix

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 7: Model Efficiency

**Objective**: Analyze computational efficiency across models.

**Metrics**:
- Number of parameters
- Model file size (MB)
- Average inference latency (ms)
- Throughput (images/sec)

**Output**:
- Performance vs Parameters scatter plot
- Performance vs Inference Time scatter plot
- Efficiency comparison table

**Results**: *[GENERATED AFTER TRAINING]*

---

## Experiment 8: Ensemble (Optional)

**Objective**: Test if combining models improves performance.

**Setup**:
- Method: Soft voting (probability averaging)
- Models: Top 2-4 performing models
- Compare ensemble vs best individual model

**Output**: Comparison table, metrics report

**Results**: *[GENERATED AFTER TRAINING]*
