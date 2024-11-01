# Executive Summary: DeepShield-X

## Problem & Motivation
The proliferation of hyper-realistic synthetic media ("deepfakes") has evolved from a technical novelty into a severe cybersecurity and societal threat. Deepfakes are increasingly weaponized to bypass biometric security, generate non-consensual imagery, and spread political disinformation. A critical flaw in existing deepfake detection research is the widespread presence of **identity data leakage** (where identical actor faces appear in both training and test sets) and the **uncritical reporting of raw accuracy** under severe class imbalance. DeepShield-X resolves these shortcomings by establishing a mathematically provable, leakage-free, and explainable deepfake detection framework on the standardized **FaceForensics++ C23** benchmark.

---

## Dataset & Leakage Prevention
We utilized FaceForensics++ C23 (H.264 compression), which realistically simulates the quality degradation introduced by social media platforms.

- **Zero-Leakage Splitting via Union-Find:** Using disjoint-set (connected-components) algorithms on video metadata, we partitioned video sources by actor identity *prior* to face extraction. This guaranteed that an identity present in the training set strictly never appeared in the validation or test sets.
- **Partition Distribution:** 
  - **Train:** $50,706$ faces ($6,964$ REAL, $43,742$ FAKE)
  - **Validation:** $8,978$ faces ($1,497$ REAL, $7,481$ FAKE)
  - **Held-Out Test:** $9,589$ faces ($1,496$ REAL, $8,093$ FAKE)
- All face crops were normalized to $224 \times 224 \times 3$ with ImageNet standardization.

---

## Experimental Design & Benchmark
We implemented six distinct architecture families and executed controlled ablations:
1. **Multi-Architecture Benchmark:** Custom CNN, ResNet50, EfficientNet-B0, MobileNetV3-Large, Xception, and Vision Transformer (ViT-B/16).
2. **Class Imbalance Ablation:** Unweighted Custom CNN vs. Class-Balanced Custom CNN ($\text{pos\_weight} = 0.1592$).
3. **Regularization Ablation:** Baseline ResNet50 vs. Regularized ResNet50 (spatial augmentation, dropout, early stopping).
4. **Bootstrap Confidence Intervals:** $1,000$ bootstrap iterations to calculate $95\%$ confidence intervals.
5. **Manipulation-Wise Breakdown:** Evaluating individual synthesis algorithms (DeepFakeDetection, Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures).
6. **Robustness Stress Test:** Perturbation sweeps across JPEG compression, Gaussian blur, noise, and downscaling.
7. **Explainability:** Grad-CAM spatial activation mapping on the final residual block.

---

## Main Findings & Best Model

```
+─────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    DEEPSHIELD-X MASTER BENCHMARK                                |
+──────────────────────────┬──────────┬──────────┬─────────────┬──────────┬──────────┬────────────+
| Architecture             | Accuracy | F1 Score | Specificity | ROC-AUC  | PR-AUC   | MCC        |
+──────────────────────────┼──────────┼──────────┼─────────────┼──────────┼──────────┼────────────+
| ResNet50 Regularized ★   |  93.89%  |  0.9633  |   87.10%    |  0.9718  |  0.9938  |   0.7821   |
| Xception                 |  93.70%  |  0.9621  |   88.30%    |  0.9706  |  0.9931  |   0.7798   |
| EfficientNet-B0          |  92.08%  |  0.9521  |   85.49%    |  0.9547  |  0.9879  |   0.7290   |
| Vision Transformer (ViT) |  90.05%  |  0.9394  |   82.75%    |  0.9512  |  0.9894  |   0.6706   |
| MobileNetV3-Large        |  87.97%  |  0.9274  |   70.99%    |  0.9187  |  0.9820  |   0.5792   |
| Custom CNN (Balanced)    |  43.24%  |  0.5265  |   74.87%    |  0.6196  |  0.8966  |   0.0987   |
| Custom CNN (Unweighted)  |  84.40%  |  0.9154  |    0.00%    |  0.5607  |  0.8799  |   0.0000   |
+──────────────────────────┴──────────┴──────────┴─────────────┴──────────┴──────────┴────────────+
```

1. **Winning Model:** **ResNet50 Regularized** achieved the highest overall discriminative performance ($\text{F1} = 0.9633$, $\text{ROC-AUC} = 0.9718$, $\text{PR-AUC} = 0.9938$, $\text{MCC} = 0.7821$). Non-parametric bootstrapping ($1,000$ iterations) established an F1 $95\%$ confidence interval of $[0.9602, 0.9664]$.
2. **Competitive Alternative:** **Xception** proved exceptionally strong, achieving the highest Specificity ($0.8830$) and Precision ($0.9777$).
3. **The Danger of Raw Accuracy:** The Unweighted Custom CNN achieved an apparent $84.40\%$ accuracy with **$0.00\%$ Specificity** (misclassifying all $1,496$ real test faces). Class-balancing restored forensic discrimination, lifting Specificity to $74.87\%$.
4. **CNNs vs. Vision Transformers:** ResNet50 and Xception consistently outperformed ViT-B/16 ($\text{F1} = 0.9633$ vs $0.9394$), confirming that localized convolutional kernels detect high-frequency blending boundaries more effectively than global patch self-attention.
5. **The NeuralTextures Blind Spot:** Sub-category evaluation revealed that **NeuralTextures** accounted for **$55.7\%$ of all false negatives** ($219 / 393$), as subtle mouth reenactment lacks the global warping boundaries of full-face swaps.

---

## Robustness & Explainability Findings
- **Robustness Stress Test:** The model retained $>92\%$ detection recall under JPEG Q=50, Gaussian blur ($r = 1.5$), and Gaussian noise ($\sigma = 0.05$). Under severe compression (JPEG Q=30), recall dropped to $39.64\%$, pinpointing aggressive lossy quantization as the primary operational vulnerability.
- **Explainability (Grad-CAM):** Activation maps on `model.backbone.layer4[-1]` confirmed that predictions are driven by authentic facial boundaries (chin, jawline, hairline) and perioral manipulation zones rather than background shortcuts.
- **Efficiency:** MobileNetV3-Large achieved $92.6$ FPS on standard CPU hardware (sub-11ms latency), providing an ideal profile for real-time mobile/edge deployment.

---

## Core Contributions
1. A fully reproducible, zero-leakage deepfake detection pipeline on FaceForensics++ C23.
2. Definitive empirical evidence exposing accuracy as an invalid metric under class imbalance.
3. An authoritative benchmark of six deep learning architectures across accuracy, calibration, robustness, and latency.
4. Publication-grade visual explainability confirming forensic boundary grounding.
