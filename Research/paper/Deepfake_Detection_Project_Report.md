# DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
## ACADEMIC MAJOR PROJECT REPORT — 2025–2026

---

# DEEPFAKE DETECTION SYSTEM
### Project DeepShield-X: Robust and Explainable Facial Manipulation Forensics
**Evaluation of Convolutional Neural Networks and Vision Transformers under Identity-Disjoint Protocols**

| Field | Details |
| :--- | :--- |
| **Candidate Name** | [Student Name] |
| **Roll Number** | [University Roll No.] |
| **Degree** | Bachelor of Technology (B.Tech) in Computer Science & Engineering |
| **Institution** | Department of Computer Science & Engineering, Engineering College / University |
| **Supervisor / Guide** | Project Evaluation Committee & Faculty Advisors |
| **Benchmark Dataset** | FaceForensics++ (H.264 C23 Compression, 69,273 Face Crops) |
| **Evaluation Protocol** | Strictly Frozen Held-Out Test Set ($N = 9,589$) · Zero Identity Leakage · Reproducible Seed ($S = 42$) |

> ℹ️ **Empirical Research Integrity Statement:** All classification metrics, parameter footprints, inference latencies, and error breakdowns reported in this project documentation are strictly measured from actual checkpointed models evaluated on a frozen held-out test partition ($N = 9,589$). No values have been fabricated, estimated, or artificially smoothed.

---

## TABLE OF CONTENTS

1. [Abstract & Executive Summary](#1-abstract--executive-summary)
2. [Introduction & Problem Statement](#2-introduction--problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Proposed Methodology & System Architecture](#4-proposed-methodology--system-architecture)
   - 4.1 Dataset Curation & Graph Union-Find Partitioning
   - 4.2 Preprocessing & $1.3\times$ Expanded Face Extraction
   - 4.3 Evaluated Neural Backbone Architectures
   - 4.4 Optimization & Multi-Stage Regularization
5. [Novelty and Key Contributions](#5-novelty-and-key-contributions)
   - 5.1 The Five Core Scientific Novelties
   - 5.2 Summary of Methodological & Practical Contributions
   - 5.3 Comparative Novelty Matrix
6. [Results and Discussion](#6-results-and-discussion)
   - 6.1 Master Multi-Architecture Benchmark
   - 6.2 Controlled Class Imbalance & Regularization Ablations
   - 6.3 Manipulation-Wise Failure Breakdown (The NeuralTextures Blind Spot)
   - 6.4 Diagnostic Audits: Compression Cliff, Calibration & Explainability
7. [Comparative Analysis with Existing Research](#7-comparative-analysis-with-existing-research)
8. [Conclusion, Limitations & Future Scope](#8-conclusion-limitations--future-scope)
9. [References](#9-references)

---

## 1. Abstract & Executive Summary

Hyper-realistic synthetic facial media ("deepfakes") generated via autoencoders, generative adversarial networks (GANs), and diffusion models have rendered visual authenticity verification acutely difficult, posing profound challenges to digital forensics, courtroom evidence, journalism, and biometric authentication. Although modern deep neural networks frequently claim $>95\%$ detection accuracy on academic benchmarks, these reported scores frequently suffer from two severe methodological flaws:
1. **Identity leakage between training and testing partitions**, where models memorize facial biometric features rather than forgery boundaries; and
2. **Extreme class imbalance**, which deceitfully inflates raw accuracy metrics.

In this project (**DeepShield-X**), we design, implement, and benchmark a complete deepfake detection framework on the compressed FaceForensics++ (FF++ C23) dataset. Sequence splitting is formulated as a graph connected-components problem solved via disjoint-set **Union-Find**, formally guaranteeing zero identity leakage across 69,273 standardized facial crops (train = 50,706; validation = 8,978; held-out test = 9,589).

We systematically compare six neural architectures: a bespoke 4-stage Custom CNN, ResNet50, Xception, EfficientNet-B0, MobileNetV3-Large, and Vision Transformer (ViT-B/16). A controlled ablation confirms that an unweighted Custom CNN collapses to predicting the majority fake class, yielding **84.40% accuracy but 0.00% specificity** (failing on every real face). Implementing positive-class weighting ($w_{\text{pos}} = 0.1592$) and multi-stage regularization enables **ResNet50 Regularized** to achieve champion performance ($F_1 = 0.9633$, ROC-AUC = 0.9718, PR-AUC = 0.9938, Specificity = 87.10%, MCC = 0.7821), closely followed by **Xception** ($F_1 = 0.9621$, Specificity = 88.30%). Sub-category audits expose NeuralTextures as the primary failure mode (55.7% of all false negatives). Probabilistic calibration ($\text{ECE} = 4.18\%$) and Grad-CAM heatmaps confirm predictions are grounded in authentic jawline and perioral manipulation boundaries.

---

## 2. Introduction & Problem Statement

Creating convincing manipulated facial imagery previously required specialized digital VFX artists and extensive computation. Today, automated synthesis pipelines can generate deceptive face swaps and reenactments in minutes. The societal risks are profound: financial fraud via synthetic executive video calls, defamatory media targeting private citizens, fabricated geopolitical announcements, and spoofed biometric face-recognition check-ins. To build trustworthy defense mechanisms, automated forensic classifiers must analyze high-frequency spatial gradients and localized blending boundaries.

However, our systematic review of current deepfake detection research reveals four critical scientific bottlenecks:

1. **Identity & Subject Leakage:** Benchmark datasets apply multiple manipulation algorithms to a shared pool of original videos. When data is split randomly or naively by video files, the same actor's face appears in both training and test subsets. The network simply memorizes facial identities rather than learning synthesis artifacts, causing up to a 10% artificial performance inflation (Dong et al., 2023).
2. **Pathological Class Imbalance:** Forensic benchmarks contain roughly 5 to 6 manipulated videos for every authentic video. Under standard unweighted cross-entropy loss, an uninformative model that trivially outputs "fake" for every input achieves 84.40% accuracy while missing 100% of authentic faces.
3. **Black-Box Decision Opacity:** Standard classifiers output a single probability score without visual grounding, preventing forensic analysts from verifying whether predictions stem from genuine face-boundary artifacts or background image shortcuts.
4. **Social Media Compression Cliff:** Platforms apply aggressive lossy compression (H.264, JPEG) that severely attenuates high-frequency forensic signals, causing steep detector degradation under low-quality transmissions.

---

## 3. Project Objectives

To resolve these bottlenecks, DeepShield-X was engineered to achieve six concrete objectives:

| Obj ID | Target Milestone | Technical Approach / Engineering Implementation | Status |
| :---: | :--- | :--- | :---: |
| **OBJ-1** | Identity-Safe Splitting Protocol | Build bipartite video graph; use Disjoint-Set Union-Find for atomic zero-leakage splits | **Verified** |
| **OBJ-2** | Multi-Architecture Benchmark | Benchmark 6 architectures across CNN, depthwise separable, compound, and ViT paradigms | **Complete** |
| **OBJ-3** | Class Imbalance Resolution | Analytically derive BCE stationary point; implement loss weighting $w_{\text{pos}} = 0.1592$ | **Complete** |
| **OBJ-4** | Granular Failure Audits | Evaluate error distributions across 6 distinct manipulation algorithms on 9,589 test crops | **Complete** |
| **OBJ-5** | Reliability & Explainability | Compute 10-bin Expected Calibration Error (ECE) and extract layer-wise Grad-CAM heatmaps | **Complete** |
| **OBJ-6** | Perturbation Stress-Testing | Evaluate accuracy retention across JPEG sweeps (Q=90 to 30), noise, blur, and scale | **Complete** |

---

## 4. Proposed Methodology & System Architecture

```
┌────────────────────┐     ┌───────────────────────┐     ┌─────────────────────────┐
│   Raw C23 Video    │ ──► │  Union-Find Splitting │ ──► │ Standardized Face Crops │
│ (FaceForensics++)  │     │ (Zero Identity Leak)  │     │     (224 × 224 × 3)     │
└────────────────────┘     └───────────────────────┘     └─────────────────────────┘
                                                                      │
                                                                      ▼
┌────────────────────┐     ┌───────────────────────┐     ┌─────────────────────────┐
│ Binary Forensic    │ ◄── │ Spatial Explainability │ ◄── │   Transfer Learning     │
│ Prediction & Audit │     │  (Grad-CAM Saliency)  │     │  (Backbone Architecture)│
└────────────────────┘     └───────────────────────┘     └─────────────────────────┘
```

### 4.1 Dataset Curation & Graph Union-Find Partitioning
We utilize the FaceForensics++ (FF++ C23) benchmark, containing 1,000 authentic pristine video sequences manipulated by four primary algorithms (Deepfakes, Face2Face, FaceSwap, NeuralTextures), alongside FaceShifter and DeepFakeDetection. The H.264 C23 ($\text{CRF} = 23$) tier is chosen because its moderate compression accurately mimics modern video distribution platforms (YouTube, Twitter, WhatsApp).

To enforce zero identity leakage, we construct an undirected bipartite graph $G = (V, E)$ where vertices $V$ denote unique video identifiers and edges $E$ represent manipulation donor-target pairings. Disjoint-Set Union-Find with path compression and union-by-rank groups all transitively linked videos into isolated connected components. Entire components are assigned atomically to training, validation, or test partitions, guaranteeing that no tracked actor's face crosses split boundaries:
$$\text{Identities}(\mathcal{D}_{\text{train}}) \cap \text{Identities}(\mathcal{D}_{\text{test}}) = \emptyset$$

| Partition | Authentic (Real) | Manipulated (Fake) | Total Face Crops | Class Ratio (F:R) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set (Train)** | 6,964 (13.7%) | 43,742 (86.3%) | **50,706** (73.2%) | 6.28 : 1 |
| **Validation Set (Val)** | 1,497 (16.7%) | 7,481 (83.3%) | **8,978** (13.0%) | 5.00 : 1 |
| **Frozen Test Set (Test)** | 1,496 (15.6%) | 8,093 (84.4%) | **9,589** (13.8%) | 5.41 : 1 |
| **Total Verified Data** | **9,957** (14.4%) | **59,316** (85.6%) | **69,273** (100.0%) | **5.96 : 1** |

### 4.2 Preprocessing & $1.3\times$ Expanded Face Extraction
Video frames are extracted at a uniform stride of 10 to suppress redundant temporal autocorrelation. Facial landmarks are detected, and bounding boxes are expanded by a scale factor of $\alpha = 1.30$ (30% perimeter expansion):
$$x_{\text{min}}' = x_c - 1.30 \frac{w}{2}, \quad y_{\text{min}}' = y_c - 1.30 \frac{h}{2}$$
This expansion is mathematically crucial: standard tight crops eliminate the jawline, chin, and forehead borders where Poisson image blending seams and color interpolation discontinuities reside. Crops are resized to $224 \times 224 \times 3$ using bilinear interpolation and standardized with ImageNet RGB parameters ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

### 4.3 Evaluated Neural Backbone Architectures
To study contrasting inductive biases, we benchmark six models:
1. **Custom CNN:** 4 sequential Conv-BatchNorm-ReLU-MaxPool stages (93,825 parameters, trained entirely from scratch).
2. **ResNet50:** 23.5M parameters, deep residual network with bottleneck blocks ($1\times 1 \to 3\times 3 \to 1\times 1$) and skip connections.
3. **Xception:** 20.8M parameters, depthwise separable convolutions decoupling spatial and cross-channel color correlation across Entry, Middle, and Exit flows.
4. **EfficientNet-B0:** 4.0M parameters, NAS-optimized compound scaling balancing depth, width, and resolution with squeeze-and-excitation attention.
5. **MobileNetV3-Large:** 1.5M parameters, hardware-aware edge backbone utilizing lightweight inverted bottlenecks and hard-swish non-linearities.
6. **ViT-B/16:** 21.7M parameters, $14 \times 14 = 196$ non-overlapping patch self-attention tokens projected across 12 transformer encoder layers.

### 4.4 Optimization & Multi-Stage Regularization
Models output a scalar logit $z \in \mathbb{R}$. To prevent majority collapse under the 6.28:1 ratio, we apply positive-class loss weighting in binary cross-entropy with logits:
$$w_{\text{pos}} = \frac{N_{\text{real}}}{N_{\text{fake}}} = \frac{6,964}{43,742} \approx 0.159206$$
Optimization uses AdamW ($\eta_0 = 10^{-4}, \lambda = 10^{-4}$) with cosine annealing preceded by a 2-epoch linear warmup. Regularization includes random horizontal flips ($p=0.5$), rotation ($\pm 15^\circ$), brightness/contrast jitter ($\pm 0.2$), dropout ($p=0.50$), and checkpointing strictly on validation ROC-AUC (patience = 5). Gaussian blur and JPEG augmentation are intentionally omitted during training to protect subtle forensic edge cues.

---

## 5. Novelty and Key Contributions

> 💡 **Distinguishing Genuine Novelty from Standard Engineering:** Merely applying transfer learning, running standard CNNs/Transformers, or loading the public FaceForensics++ benchmark does NOT constitute scientific novelty. DeepShield-X contributes five distinct, verifiable methodological and empirical innovations designed to solve systemic vulnerabilities identified in prior published forensic literature.

### 5.1 The Five Core Scientific Novelties

1. **Graph-Connected Disjoint-Set Identity Isolation Protocol:**
   Prior studies frequently employ naive video-level or random frame splitting, which allows identical actors to appear in both train and test partitions through manipulation donor-target pairings. We formulated dataset partitioning as a graph connected-components problem solved via Disjoint-Set Union-Find with path compression and union-by-rank. By assigning entire multi-edge clusters atomically, we provide a mathematically verifiable guarantee of zero identity leakage across 69,273 face crops, ensuring models learn genuine forgery artifacts rather than facial biometric memorization.
2. **Analytical Derivation & Controlled Empirical Proof of Imbalance Collapse:**
   We mathematically derived from the BCE gradient expectation stationary condition ($\mathbb{E}[\partial \mathcal{L} / \partial z] = \sigma(z) - \pi_1 = 0$) that an unweighted model optimizes training risk by producing a constant positive logit ($z^* = +1.837$), causing complete majority-class collapse. We then proved this experimentally: an unweighted CNN achieves 84.40% accuracy but exactly 0.00% specificity (fails on every real face). This concrete proof demonstrates why raw accuracy is an invalid forensic metric and establishes MCC as the primary evaluation standard.
3. **Discovery & Attribution of the NeuralTextures Blind Spot:**
   Through granular sub-category auditing across 9,589 frozen test samples, we uncovered that NeuralTextures accounts for 55.7% (219 of 393) of all false negatives across the entire benchmark. We identified the root cause: deferred neural rendering modifies only the mouth cavity, and Global Average Pooling (GAP) in standard backbones dilutes these localized signals across the vast authentic face. This isolates a concrete architectural deficiency in standard vision backbones for facial reenactment detection.
4. **Empirical Quantification of the JPEG Compression Cliff:**
   We performed multi-tier perturbation stress-testing ($N = 502$) mapping accuracy retention across compression rates. While spatial models remain resilient to additive noise (99.4%) and blur (100%), we identified a non-linear performance cliff below JPEG Quality 50, where recall collapses from 96.41% (Q=70) to 39.64% (Q=30). This quantifies the exact boundary where lossy quantization destroys high-frequency Poisson boundary gradients, defining operational limits for social media screening.
5. **Explainability & Probabilistic Calibration Grounding:**
   Unlike black-box classifiers, DeepShield-X integrates 10-bin Expected Calibration Error analysis (achieving $\text{ECE} = 4.18\%$, Brier = 0.0524) demonstrating near-perfect reliability at extreme operational bins ($\text{gap} < 0.008$). Layer-wise Grad-CAM heatmaps verify that ResNet50 attends specifically to facial boundary seams for face-swap fakes and the perioral cavity for reenactments, proving forensic grounding.

---

### 5.2 Summary of Methodological & Practical Contributions
In summary, DeepShield-X delivers an end-to-end, leakage-free forensic benchmark rather than an incremental model tuning exercise. By mathematically enforcing zero identity overlap across 69,273 crops, resolving severe class-imbalance metric distortion, exposing the perioral pooling vulnerability in facial reenactments, and quantifying compression breakdown boundaries, the proposed framework provides reproducible, defensible empirical benchmarks for practical forensic deployment.

---

### 5.3 Comparative Novelty Matrix

| Evaluation Dimension | Standard Baselines | ID-Reveal (2021) | SBI (2022) | Dong et al. (2023) | DeepfakeBench (2023) | **DeepShield-X (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Identity-Disjoint Split** | Random (Leaky) | Partial Video | Synthetic Pairs | Verified Disjoint | Video-level Only | **Graph Union-Find Verified** |
| **Imbalance Derivation** | None (Ignored) | None | None | None | None | **Mathematical Proof + Ablation** |
| **Multi-Architecture Suite** | Single Model | Single Backbone | Single Backbone | ResNet / Xception | 15+ Baselines | **6 Architectures (CNN to ViT)** |
| **Sub-Category Failure Audit** | None | None | None | None | Partial | **Granular 6-Family Breakdown** |
| **Compression Cliff Profile** | None | None | H.264 Only | None | H.264 Tiers | **JPEG Sweeps (Q=90 to 30)** |
| **Spatial Explainability** | Black-Box | Temporal Emb. | Mask Blending | Feature Norm. | AUC Only | **Grad-CAM + 10-Bin ECE** |

---

## 6. Results and Discussion

### 6.1 Master Multi-Architecture Benchmark
All models were evaluated on the strictly frozen, held-out test partition ($N = 9,589$; Authentic Real = 1,496; Manipulated Fake = 8,093) under a fixed decision threshold $\tau = 0.50$ without test-set tuning.

| Architecture | Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | PR-AUC | Bal. Acc. | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 Regularized ★** | **0.9389** | 0.9755 | **0.9514** | 0.8710 | **0.9633** | **0.9718** | **0.9938** | 0.9112 | **0.7821** |
| **Xception** | 0.9370 | **0.9777** | 0.9470 | **0.8830** | 0.9621 | 0.9706 | 0.9931 | **0.9150** | 0.7798 |
| **EfficientNet-B0** | 0.9208 | 0.9721 | 0.9330 | 0.8549 | 0.9521 | 0.9547 | 0.9879 | 0.8940 | 0.7290 |
| **Vision Transformer (ViT-B/16)** | 0.9005 | 0.9663 | 0.9140 | 0.8275 | 0.9394 | 0.9512 | 0.9894 | 0.8708 | 0.6706 |
| **MobileNetV3-Large** | 0.8797 | 0.9444 | 0.9110 | 0.7099 | 0.9274 | 0.9187 | 0.9820 | 0.8105 | 0.5792 |
| *ResNet50 Baseline (Unregularized)* | *0.9133* | *0.9709* | *0.9250* | *85.03%* | *0.9474* | *0.9578* | *0.9914* | *0.8876* | *0.7087* |
| *Custom CNN (Class-Balanced)* | *0.4324* | *0.8895* | *0.3739* | *74.87%* | *0.5265* | *0.6196* | *0.8966* | *0.5613* | *0.0987* |
| *Custom CNN (Unweighted Baseline)* | *0.8440* | *0.8440* | *1.0000* | *0.0000* | *0.9154* | *0.5607* | *0.8799* | *0.5000* | *0.0000* |

### 6.2 Controlled Class Imbalance & Regularization Ablations
- **Class Imbalance Ablation:** The Unweighted Custom CNN achieves an apparent accuracy of 84.40% and F1 of 0.9154, but possesses Specificity = 0.0000 and MCC = 0.0000. It produces 8,093 True Positives and 1,496 False Positives, classifying every authentic face as fake. Implementing positive-class weighting ($w_{\text{pos}} = 0.1592$) forces gradient equilibrium, restoring specificity to 74.87%.
- **Regularization Ablation:** On ResNet50, multi-stage regularization (spatial augmentation, dropout 0.50, ROC-AUC early stopping) improves Accuracy by $+2.56\%$, F1 by $+1.59\%$, Specificity by $+2.07\%$, and MCC by $+7.34\%$ ($0.7087 \to 0.7821$), proving regularization prevents identity memorization.

### 6.3 Manipulation-Wise Failure Breakdown (The NeuralTextures Blind Spot)

| Manipulation Category | Ground Truth | Test Samples | Recall / Specificity | Mean Pred. P(fake) | Errors | Error Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepFakeDetection** | FAKE | 615 | **99.67%** (Recall) | 0.9889 | FN = 2 | 0.5% |
| **FaceSwap** | FAKE | 1,496 | 97.33% (Recall) | 0.9658 | FN = 40 | 10.2% |
| **Deepfakes** | FAKE | 1,492 | 97.32% (Recall) | 0.9657 | FN = 40 | 10.2% |
| **Face2Face** | FAKE | 1,499 | 97.00% (Recall) | 0.9622 | FN = 45 | 11.5% |
| **FaceShifter** | FAKE | 1,495 | 96.86% (Recall) | 0.9568 | FN = 47 | 12.0% |
| **NeuralTextures (Blind Spot)** | FAKE | 1,496 | **85.36%** (Recall) | 0.8363 | FN = **219** | **55.7% !** |
| **Original (Authentic Real)** | REAL | 1,496 | **87.10%** (Specificity) | 0.1497 | FP = 193 | — |

> ⚠️ **The NeuralTextures Finding:** Out of 393 total false negatives across all 9,589 test samples, **219 (55.7%) occurred on NeuralTextures**. Because NeuralTextures limits facial reenactment strictly to the mouth interior, standard Global Average Pooling aggregates oral anomalies with the remaining 90% unedited facial skin, severely attenuating the discriminative gradient signal.

### 6.4 Diagnostic Audits: Compression Cliff, Calibration & Explainability
1. **CNNs Outperform Vision Transformers:** ResNet50 ($F_1 = 0.9633$) and Xception ($F_1 = 0.9621$) substantially outperformed ViT-B/16 ($F_1 = 0.9394$). Facial manipulation boundaries (Poisson stitching seams, blending artifacts) are local, high-frequency spatial gradients. The translation equivariance of convolutional kernels naturally captures these sharp edges, whereas ViT's fixed $16\times 16$ patch projection tokenizes and fragments continuous boundary lines across adjacent patch borders.
2. **Perturbation Robustness & JPEG Cliff:** Testing on $N=502$ synthetic crops revealed stability under Gaussian blur (100% recall), additive Gaussian noise (99.4%), and $4\times$ downscale-upscale cycles (100%). However, performance plunged sharply from 96.41% at JPEG Q=70 to 72.51% at Q=50, collapsing to 39.64% at Q=30, indicating spatial detectors depend on fine-grained DCT coefficients eliminated by heavy quantization.
3. **Calibration & Grad-CAM:** ResNet50 achieves Expected Calibration Error ($\text{ECE} = 4.18\%$) and Brier score = 0.0524. Over 80% of test predictions reside in the $[0.9, 1.0]$ confidence bin with a calibration gap of only 0.0074. Grad-CAM confirms attention focuses on the jawline and chin for face-swaps and perioral regions for reenactments, verifying predictions are anchored in physical manipulation artifacts rather than spurious background cues.

---

## 7. Comparative Analysis with Existing Research

| Model / Paper | Year | Detection Paradigm | Benchmark | Core Strength | Primary Limitation | How DeepShield-X Relates / Differs |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **ID-Reveal [7]** | 2021 | Temporal Biometric Representation | FF++ (C23, C40) | Robust to facial reenactment; identity-aware | Requires temporal video; partial identity split | We enforce formal graph Union-Find disjointness on static single frames |
| **Multi-Attn [8]** | 2021 | Region Multi-Attention CNN | FF++, Celeb-DF | Attends to localized facial components | High computational cost; no imbalance analysis | We diagnose why GAP causes failure on NeuralTextures and benchmark 6 models |
| **LipForensics [9]** | 2021 | Spatio-temporal Mouth Dynamics | FF++, DFDC | High generalization across unseen face swaps | Limited to speaking faces; misses static images | We evaluate full-face boundary cues and reveal oral cavity failure under static pooling |
| **SBI [10]** | 2022 | Self-Blended Boundary Augmentation | FF++, CDF, DFD | Superior out-of-distribution transfer | Synthesizes synthetic artifacts; no identity audit | We benchmark empirical architectures under real-world C23 social media compression |
| **Cross-Eff-ViT [11]** | 2022 | EfficientNet + ViT Hybrid | FF++, DFDC | Combines local CNN & global ViT tokens | High parameter count; untested class imbalance | We isolate CNN vs ViT backbones directly and prove pure CNNs lead on C23 media |
| **Dong et al. [12]** | 2023 | Implicit Identity Leakage Proof | FF++, Celeb-DF | Formal mathematical proof of leakage bias | Focuses only on proof; no full forensic suite | We solve the leakage problem via graph Union-Find and benchmark 6 architectures |
| **DeepfakeBench [13]** | 2023 | Unified Detection Benchmark | FF++, CDF, DFDC | Standardized benchmark of 15+ detectors | Uses naive video splits; no imbalance derivation | We add Union-Find identity isolation and provide explicit class imbalance proofs |
| **DeepShield-X (Ours)** | **2026** | **Identity-Disjoint Forensic Suite** | **FF++ C23 (69K)** | **Unified 6-dimension experimental investigation** | **Frame-level scope; single benchmark evaluation** | **First unified framework combining Union-Find, imbalance proof, & 6-arch suite** |

---

## 8. Conclusion, Limitations & Future Scope

### 8.1 Conclusion & Deployment Recommendations
DeepShield-X delivers an academically rigorous, reproducible deepfake detection framework. By applying graph Union-Find, we eliminated identity leakage across 69,273 FF++ C23 crops. We proved mathematically and experimentally that raw accuracy masks catastrophic failure under class skew, while positive-class weighting enables **ResNet50 Regularized** ($F_1 = 0.9633$, ROC-AUC = 0.9718) and **Xception** ($F_1 = 0.9621$) to lead detection performance. **MobileNetV3-Large** (35.4 FPS, 17.6 MB) provides a viable edge deployment option. Spatial heatmaps and calibration metrics confirm predictions are anchored in authentic boundary anomalies.

| Deployment Scenario | Recommended Model | Practical Rationale |
| :--- | :--- | :--- |
| **Forensic Investigation & Legal Evidence** | ResNet50 Regularized | Highest $F_1$ (0.9633), ROC-AUC (0.9718), MCC (0.7821) |
| **Commercial Platform Moderation (High Throughput)** | Xception | Leading Specificity (88.30%) and Precision (97.77%) |
| **Mobile & Edge Device Verification** | MobileNetV3-Large | 35.4 FPS real-time throughput, 17.6 MB checkpoint footprint |

### 8.2 Honest Limitations & Future Scope
1. **Cross-Dataset Scope:** Current evaluation is restricted to FaceForensics++ C23; future work will test out-of-distribution transfer to Celeb-DF, DFDC, and diffusion-generated faces.
2. **Temporal Integration:** Operating on static crops misses inter-frame lip-sync jitter; future extensions will integrate spatio-temporal LSTM/Transformer heads.
3. **Perioral Attention:** Dedicated regional attention sub-networks will be engineered to resolve the NeuralTextures blind spot (55.7% FN).
4. **Frequency-Domain Fusion:** Combining spatial features with DCT/DTCWT spectral analysis will mitigate detector collapse below the JPEG Q=50 compression cliff.

---

## 9. References

1. A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Niessner, "FaceForensics++: Learning to detect manipulated facial images," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1–11.
2. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
3. F. Chollet, "Xception: Deep learning with depthwise separable convolutions," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1251–1258.
4. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021.
5. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.
6. D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," *BMC Genomics*, vol. 21, no. 6, 2020.
7. D. Cozzolino, G. Poggi, J. Thies, M. Niessner, and L. Verdoliva, "ID-Reveal: Identity-aware deepfake video detection," in *Proc. IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 15108–15117.
8. H. Zhao, W. Zhou, D. Chen, T. Wei, W. Zhang, and N. Yu, "Multi-attentional deepfake detection," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 2185–2194.
9. A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, "Lips don't lie: A generalisable approach to face forgery detection," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 5039–5049.
10. K. Shiohara and T. Yamasaki, "Detecting deepfakes with self-blended images," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 18720–18729.
11. I. Coccomini, N. Messina, C. Gennaro, and F. Falchi, "Combining EfficientNet and Vision Transformers for video deepfake detection," in *Proc. International Conference on Image Analysis and Processing (ICIAP)*, 2022, pp. 219–229.
12. S. Dong, J. Wang, J. Liang, H. Fan, and R. Ji, "Implicit identity leakage: The stumbling block to improving deepfake detection generalization," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 3994–4004.
13. Z. Yan, Y. Zhang, Y. Fan, and B. Wu, "DeepfakeBench: A comprehensive benchmark of deepfake detection," in *NeurIPS Datasets and Benchmarks Track*, 2023.

---
*DeepShield-X Academic Project Documentation · Department of Computer Science & Engineering · All Rights Reserved.*
