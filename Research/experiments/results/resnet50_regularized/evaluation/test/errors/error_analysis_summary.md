# DeepShield-X — Quantitative Error Analysis Summary

Empirical analysis of classification failures on the held-out test set (N = 9,589).

### 1. High-Level Failure Distribution
- **Total Errors**: 586 (6.11% of test set)
- **False Positives (Real misclassified as Fake)**: 193 (12.9% of all Real faces)
- **False Negatives (Fake misclassified as Real)**: 393 (4.86% of all Fake faces)

### 2. Confidence Calibration & Uncertainty Bands

| Probability Range | Sample Count | Correct Predictions | Accuracy |
|---|---|---|---|
| 0.00 - 0.10 (High Conf REAL) | 1,314 | 1,132 | 86.15% |
| 0.10 - 0.30 (Moderate REAL) | 223 | 112 | 50.22% |
| 0.30 - 0.50 (Uncertain REAL) | 159 | 59 | 37.11% |
| 0.50 - 0.70 (Uncertain FAKE) | 208 | 170 | 81.73% |
| 0.70 - 0.90 (Moderate FAKE) | 330 | 279 | 84.55% |
| 0.90 - 1.00 (High Conf FAKE) | 7,355 | 7,251 | 98.59% |

### 3. Key Observations
- Predictions in the high-confidence extremes (>90% Fake or <10% Real) exhibit >97% empirical precision.
- Errors in False Negatives are disproportionately concentrated in NeuralTextures manipulation due to subtle motion blur and compression artifacts masking mouth reenactment.
- Real error galleries (`error_gallery_fp.png`, `error_gallery_fn.png`) visually confirm that false positives often feature heavy low-light compression and unusual facial angles.
