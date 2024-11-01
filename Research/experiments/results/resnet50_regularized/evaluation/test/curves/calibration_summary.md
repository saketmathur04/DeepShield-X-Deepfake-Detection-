# DeepShield-X — Calibration & Threshold Analysis

Reliability calibration for ResNet50 Regularized evaluated strictly on the test partition.

- **Brier Score**: `0.0499` (Close to 0 indicates excellent probabilistic alignment)
- **Expected Calibration Error (ECE)**: `0.0407` (4.07%)
- **Threshold Operating Point**: Fixed at `0.50` (selected via validation ROC-AUC optimization without test-set tuning).

### Calibration Reliability Table

| Confidence Bin | Samples in Bin | Avg Predicted Probability | Empirical Accuracy | Calibration Gap |
|---|---|---|---|---|
| [0.0, 0.1] | 1,314 | 0.0150 | 0.1385 | 0.1235 |
| [0.1, 0.2] | 129 | 0.1463 | 0.5194 | 0.3730 |
| [0.2, 0.3] | 94 | 0.2505 | 0.4681 | 0.2176 |
| [0.3, 0.4] | 76 | 0.3461 | 0.6579 | 0.3118 |
| [0.4, 0.5] | 83 | 0.4553 | 0.6024 | 0.1471 |
| [0.5, 0.6] | 92 | 0.5527 | 0.7935 | 0.2408 |
| [0.6, 0.7] | 116 | 0.6494 | 0.8362 | 0.1868 |
| [0.7, 0.8] | 130 | 0.7505 | 0.7692 | 0.0187 |
| [0.8, 0.9] | 200 | 0.8570 | 0.8950 | 0.0380 |
| [0.9, 1.0] | 7,355 | 0.9953 | 0.9859 | 0.0094 |
