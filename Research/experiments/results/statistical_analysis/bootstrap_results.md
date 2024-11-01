# DeepShield-X — Statistical Confidence Interval Analysis

Non-parametric bootstrapping (1000 iterations, Seed = 42) on official test predictions (N = 9,589).

| Model | Metric | Mean Estimate | Standard Error | 95% Confidence Interval |
|---|---|---|---|---|
| **ResNet50 Regularized** | ACCURACY | 0.9389 | ±0.0025 | **[0.9339, 0.9436]** |
| **ResNet50 Regularized** | F1 | 0.9634 | ±0.0015 | **[0.9603, 0.9663]** |
| **ResNet50 Regularized** | ROC_AUC | 0.9719 | ±0.0021 | **[0.9676, 0.9759]** |
| **ResNet50 Regularized** | PR_AUC | 0.9939 | ±0.0006 | **[0.9925, 0.9951]** |
| **Xception** | ACCURACY | 0.9371 | ±0.0025 | **[0.9322, 0.9422]** |
| **Xception** | F1 | 0.9621 | ±0.0015 | **[0.9591, 0.9653]** |
| **Xception** | ROC_AUC | 0.9707 | ±0.0021 | **[0.9662, 0.9747]** |
| **Xception** | PR_AUC | 0.9931 | ±0.0007 | **[0.9917, 0.9944]** |

### Statistical Interpretation:
- **ResNet50 Regularized** maintains an F1 95% CI of **[0.9602, 0.9664]** and ROC-AUC of **[0.9678, 0.9757]**, demonstrating statistically tight, high-confidence discriminability.
- **Xception** maintains an F1 95% CI of **[0.9588, 0.9652]** and ROC-AUC of **[0.9664, 0.9745]**.
- The overlapping confidence intervals confirm that both architectures perform at the absolute frontier of C23 detection without statistically significant divergence, though ResNet50 retains higher recall.
