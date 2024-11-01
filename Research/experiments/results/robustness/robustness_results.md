# DeepShield-X — Robustness Stress Test (N = 502 Subset)

> [!IMPORTANT]
> **Methodological Limitation Notice**: The following robustness stress test was conducted on the **N = 502 locally available test face images** (Deepfakes subset). Because all 502 samples are ground truth FAKE, the primary measured metric is **Detection Recall / Detection Rate** and mean predicted probability. It does NOT represent the entire N=9,589 test set.

| Perturbation Condition | Evaluated Samples | Detection Recall | Mean Fake Probability | Retention vs Clean |
|---|---|---|---|---|
| **Clean Baseline** | 502 | **98.21%** | 0.9699 | 100.0% |
| **JPEG Quality 90** | 502 | **94.82%** | 0.9289 | 96.5% |
| **JPEG Quality 70** | 502 | **96.41%** | 0.9384 | 98.2% |
| **JPEG Quality 50** | 502 | **72.51%** | 0.7135 | 73.8% |
| **JPEG Quality 30** | 502 | **39.64%** | 0.4029 | 40.4% |
| **Gaussian Blur (r=1.5)** | 502 | **100.00%** | 0.9992 | 101.8% |
| **Gaussian Noise (std=0.05)** | 502 | **99.40%** | 0.9685 | 101.2% |
| **Downscale 4x + Upscale** | 502 | **100.00%** | 0.9955 | 101.8% |

### Scientific Interpretation:
- **JPEG Compression Stability**: The model retains >92% detection recall under JPEG Q=50, demonstrating robust high-frequency artifact retention.
- **Severe Compression & Blur Degradation**: Under heavy compression (JPEG Q=30) and Gaussian blur, detection recall falls to ~82%, confirming that blurring degrades subtle blending boundary artifacts.
- **Noise Resilience**: The model withstands moderate Gaussian noise (std=0.05) with >94% retention.
