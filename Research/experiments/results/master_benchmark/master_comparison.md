# DeepShield-X — Master Model Benchmark Comparison

Authoritative evaluation on the official held-out test set (N = 9,589; REAL = 1,496, FAKE = 8,093).
Threshold = 0.50. All models evaluated under identical leakage-free protocol.

## 1. Primary Architecture & Controlled Ablation Benchmark

| Architecture | Accuracy | Precision | Recall (Sens.) | Specificity | F1 Score | ROC-AUC | PR-AUC | Bal. Acc. | MCC | Cohen's Kappa |
|---|---|---|---|---|---|---|---|---|---|---|
| **Custom CNN (Unweighted Baseline)** | 0.8440 | 0.8440 | 1.0000 | 0.0000 | **0.9154** | 0.5607 | 0.8799 | 0.5000 | 0.0000 | 0.0000 |
| **Custom CNN (Class-Balanced)** | 0.4324 | 0.8895 | 0.3739 | 0.7487 | **0.5265** | 0.6196 | 0.8966 | 0.5613 | 0.0000 | 0.0000 |
| **ResNet50 Baseline** | 0.9133 | 0.9709 | 0.9250 | 0.8503 | **0.9474** | 0.9578 | 0.9914 | 0.8876 | 0.7087 | 0.7020 |
| **ResNet50 Regularized (Winner)** | 0.9389 | 0.9755 | 0.9514 | 0.8710 | **0.9633** | 0.9718 | 0.9938 | 0.9112 | 0.7821 | 0.7799 |
| **EfficientNet-B0** | 0.9208 | 0.9721 | 0.9330 | 0.8549 | **0.9521** | 0.9547 | 0.9879 | 0.8940 | 0.7290 | 0.7239 |
| **MobileNetV3-Large** | 0.8797 | 0.9444 | 0.9110 | 0.7099 | **0.9274** | 0.9187 | 0.9820 | 0.8105 | 0.5792 | 0.5760 |
| **Xception** | 0.9370 | 0.9777 | 0.9470 | 0.8830 | **0.9621** | 0.9706 | 0.9931 | 0.9150 | 0.7798 | 0.7763 |
| **Vision Transformer (ViT-B/16)** | 0.9005 | 0.9663 | 0.9140 | 0.8275 | **0.9394** | 0.9512 | 0.9894 | 0.8708 | 0.6706 | 0.6625 |

## 2. Full Error Matrix & Diagnostic Breakdown

| Architecture | TP | TN | FP | FN | NPV | FPR | FNR |
|---|---|---|---|---|---|---|---|
| **Custom CNN (Unweighted Baseline)** | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| **Custom CNN (Class-Balanced)** | 0 | 0 | 0 | 0 | 0.0000 | 0.2513 | 0.6261 |
| **ResNet50 Baseline** | 7486 | 1272 | 224 | 607 | 0.6770 | 0.1497 | 0.0750 |
| **ResNet50 Regularized (Winner)** | 7700 | 1303 | 193 | 393 | 0.7683 | 0.1290 | 0.0486 |
| **EfficientNet-B0** | 7551 | 1279 | 217 | 542 | 0.7024 | 0.1451 | 0.0670 |
| **MobileNetV3-Large** | 7373 | 1062 | 434 | 720 | 0.5960 | 0.2901 | 0.0890 |
| **Xception** | 7664 | 1321 | 175 | 429 | 0.7549 | 0.1170 | 0.0530 |
| **Vision Transformer (ViT-B/16)** | 7397 | 1238 | 258 | 696 | 0.6401 | 0.1725 | 0.0860 |
