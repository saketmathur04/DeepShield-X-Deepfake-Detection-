# Robust and Explainable DeepFake Detection Using Identity-Disjoint Evaluation of Convolutional Neural Networks and Vision Transformers

**Author:** Saket Mathur  
**Affiliation:** Department of Computer Science and Engineering  
**Evaluation Protocol:** Frozen Held-Out Partition ($N = 9,589$), Zero Shared Tracked Identifiers, Disjoint-Set Union-Find Splitting, Reproducible Seed Protocol ($S = 42$)  
**Target Format:** IEEE Conference Format (Academic Research Manuscript)  

---

## Abstract

The synthesis of hyper-realistic facial manipulations (deepfakes) introduces severe challenges to digital media forensics, information integrity, and biometric authentication. Despite rapid advances in automated detection architectures, forensic evaluation protocols frequently suffer from critical methodological vulnerabilities: identity and source sequence leakage between training and evaluation partitions, and the deceptive inflation of conventional classification metrics under severe class imbalance. In this study, we present a systematic experimental investigation of deepfake detection on the compressed FaceForensics++ (FF++ C23) benchmark under a strictly verified identity-disjoint partitioning protocol. We formulate sequence partitioning as a graph connected-components problem solved via disjoint-set union-find, guaranteeing that no identity or source recording is shared across the training ($N=50,706$), validation ($N=8,978$), and frozen test ($N=9,589$) partitions across 69,273 standardized facial crops. We conduct a rigorous comparative evaluation of six neural architectures spanning diverse structural paradigms: a bespoke 4-stage Custom CNN, ResNet50, EfficientNet-B0, MobileNetV3-Large, Xception, and Vision Transformer (ViT-B/16). Controlled ablations demonstrate that an unweighted Custom CNN collapses toward majority-class prediction, producing an apparent accuracy of 84.40\% but 0.00\% specificity, highlighting the pathological failure of raw accuracy under prevalence skew. Incorporating empirical class weighting and comprehensive regularization enables ResNet50 to achieve leading performance across primary discriminative metrics (F1 = 0.9633, ROC-AUC = 0.9718, PR-AUC = 0.9938, Specificity = 0.8710, MCC = 0.7821), closely followed by Xception (F1 = 0.9621, Specificity = 0.8830). ViT-B/16 achieves competitive performance (F1 = 0.9394, ROC-AUC = 0.9512) but does not surpass residual and depthwise separable convolutions in this compressed image setting. Non-parametric bootstrap resampling ($B=1,000$) establishes a 95\% percentile confidence interval of $[0.9602, 0.9664]$ for ResNet50 F1. Fine-grained sub-category failure audits reveal that NeuralTextures accounts for 55.7\% (219/393) of all false negatives, reflecting the challenge of detecting subtle perioral modifications. Perturbation stress tests on an $N=502$ synthetic image subset demonstrate performance retention under moderate JPEG compression (Q=70) with sharp degradation under severe quantization (Q=30). Finally, Grad-CAM spatial activation mapping confirms that discriminative focus aligns with facial boundary and perioral regions, while highlighting the epistemological limitations of post-hoc saliency methods in multimedia forensics.

**Index Terms:** Deepfake forensics, multimedia security, identity leakage, convolutional neural networks, vision transformers, class imbalance, model calibration, visual explainability.

---

## I. Introduction

The weaponization of generative deep learning models for facial manipulation poses an unprecedented threat to digital trust, legal integrity, and national security. While initial synthetic media generation techniques produced noticeable visual flaws---such as unnatural blinking cadences, distorted facial silhouettes, and lighting discrepancies---contemporary synthesis pipelines leveraging generative adversarial networks (GANs), autoencoders, and neural radiance or diffusion models generate high-fidelity videos that effortlessly deceive human observers.

Consequently, computer vision systems have been tasked with serving as automated forensic filters across social media channels, newsrooms, and enterprise identity verification systems. However, the academic literature in deepfake forensics has recently encountered a reproducibility and generalization crisis. Detectors reporting near-perfect accuracy ($>99\%$) on academic benchmarks frequently exhibit substantial performance degradation when evaluated under unconstrained operational conditions. A rigorous examination of the forensic literature reveals four recurring methodological deficiencies:

1. **Cross-Partition Identity and Source Leakage:** Video benchmark datasets often feature multiple manipulated instances derived from a common set of authentic identity recordings. Random frame-level or video-level splits routinely assign frames of Actor $A$ to the training set and manipulated pairings involving Actor $A$ to the evaluation partition. The detector consequently memorizes facial identity geometry rather than learning generalizable boundary synthesis artifacts.
2. **Deceptive Metrics Under Class Skew:** Deepfake datasets are inherently imbalanced due to combinatorial generation, where each real video serves as the source for four or more distinct manipulation algorithms (producing typical fake-to-real ratios of $5:1$ to $6.3:1$). Minimizing unweighted cross-entropy loss incentivizes the network to collapse its decision threshold toward the majority fake class, producing deceptively high raw accuracy while rendering the system utterly incapable of validating authentic media ($\text{Specificity} \to 0.00\%$).
3. **Absence of Rigorous Statistical Uncertainty Quantification:** Many forensic studies report single-point test metrics without confidence intervals or statistical hypothesis testing, obscuring whether reported marginal differences represent genuine architectural superiority or stochastic artifact alignment.
4. **Uncalibrated Confidence and Opaque Decision Boundaries:** Deep neural networks operate as black-box function approximators. Without spatial explainability, models may trigger on compression noise, background clutter, or digital watermarks. Furthermore, the degradation of forensic clues under lossy compression (e.g., JPEG, H.264) is rarely mapped across quantitative degradation curves.

To address these vulnerabilities, this study presents a systematic experimental investigation of deepfake detection on the compressed FaceForensics++ (FF++ C23) benchmark. We implement an identity-disjoint partitioning protocol formulated as a graph connected-components problem solved via disjoint-set union-find, guaranteeing that no tracked identity or source recording is shared across the training ($N=50,706$), validation ($N=8,978$), and frozen test ($N=9,589$) partitions across 69,273 standardized facial crops.

We systematically benchmark six distinct neural architectures: a bespoke 4-stage Custom CNN, ResNet50, EfficientNet-B0, MobileNetV3-Large, Xception, and Vision Transformer (ViT-B/16). Furthermore, we conduct controlled ablations on loss weighting and multi-stage regularization, quantify uncertainty via non-parametric bootstrap resampling ($B=1,000$), perform fine-grained sub-category audits across six manipulation pipelines, evaluate probabilistic calibration via Expected Calibration Error (ECE) and Brier scores, stress-test robustness across eight digital perturbation conditions on an $N=502$ synthetic image subset, and evaluate spatial decision boundaries using Gradient-weighted Class Activation Mapping (Grad-CAM).

The primary contributions of this paper are:
- An identity-disjoint evaluation protocol based on graph connected-components partitioning via disjoint-set union-find, providing strict separation of tracked source identities across 69,273 FaceForensics++ C23 face crops.
- An empirical demonstration of metric pathology under class imbalance, showing that an unweighted Custom CNN achieves 84.40\% accuracy while collapsing to 0.00\% specificity, accompanied by an exact analytical loss derivation.
- A controlled comparative benchmark of six neural architectures spanning convolutional and transformer paradigms on compressed forensic media.
- An ablation analysis of multi-stage regularization and validation-driven model selection, demonstrating measurable performance gains on held-out test data.
- Sub-category error dissection across six manipulation pipelines identifying NeuralTextures as the primary source of false negatives (55.7\%), accompanied by calibration analysis, perturbation stress-testing, and a critical appraisal of post-hoc Grad-CAM explainability.

---

## II. Related Work

### A. Deepfake Generation and Facial Manipulation
Facial manipulation techniques in digital video are generally categorized into four primary paradigms: entire face synthesis, identity swapping, facial reenactment, and facial attribute editing. Entire face synthesis generates synthetic identities using models such as ProGAN and StyleGAN. Identity swapping replaces the facial region of a target individual with that of a source identity while preserving target pose, lighting, and expression. Classical methods such as FaceSwap use 3D facial landmarks, Delaunay triangulation, and Poisson image blending. Deepfake autoencoders employ shared encoder and identity-specific decoder networks. More recent pipelines like FaceShifter incorporate multi-level target-adaptive appearance fusion to address occlusions and complex facial poses.

Facial reenactment transfers source expressions onto target videos via RGB deformation transfer (e.g., Face2Face). More recently, deferred neural rendering methods like NeuralTextures utilize learned neural texture maps coupled with photometric reconstruction loss to manipulate lip and mouth movements while retaining background facial geometry. Because NeuralTextures restricts modifications primarily to the mouth interior and perioral region while preserving the remainder of the face, it introduces subtle photometric inconsistencies that represent a distinct forensic challenge.

### B. Deepfake Detection Methodologies
Forensic detection initially relied on handcrafted biometric signals, including abnormal blink rates, head pose inconsistencies, and corneal reflection mismatches. However, pixel-level deep learning models rapidly demonstrated superior generalization. Chollet's Xception, which factorizes standard convolutions into channel-wise spatial convolutions followed by pointwise ($1 \times 1$) cross-channel linear projections, emerged as the reference baseline in FaceForensics++. Deep residual networks (ResNet) solved gradient degradation through shortcut connections, allowing deep feature hierarchies to capture fine blending discontinuities.

To optimize computational throughput, EfficientNet scales depth, width, and resolution via compound scaling principles, while MobileNetV3 combines hardware-aware neural architecture search (NAS), depthwise convolutions, and Squeeze-and-Excitation attention modules. Vision Transformers (ViT) segment images into non-overlapping spatial patches and process them using standard transformer encoders with Multi-Head Self-Attention (MHSA). In parallel, frequency-domain forensic techniques have emerged to detect spectral anomalies left by generative up-sampling operations, while temporal models track inter-frame dynamic inconsistencies.

### C. Dataset Integrity and Partition Contamination
The establishment of rigorous forensic benchmarks has been pivotal in advancing detection capabilities, including FaceForensics++ (FF++), Celeb-DF, the DeepFake Detection Challenge (DFDC), and WildDeepfake. The C23 tier of FaceForensics++, compressed via H.264 at Constant Rate Factor 23, corresponds to typical lossy social media distribution and constitutes our primary experimental substrate.

A pervasive vulnerability in video-based deepfake forensics is actor identity leakage. When frames originating from the same authentic video appear in both training and test partitions, models achieve artificially inflated accuracy by memorizing facial geometry, hair color, or background cues rather than forensic synthesis anomalies. Cozzolino et al. emphasized the necessity of identity-disjoint splitting to guarantee that detectors learn forensic synthesis anomalies rather than biometric facial signatures.

### D. Forensic Evaluation Protocols, Imbalance, and Uncertainty
Forensic benchmarks are characterized by severe class imbalance due to combinatorial manipulation generation. Methodological literature in machine learning has demonstrated that conventional accuracy and unweighted F1 scores provide misleading assessments under skewed class prevalence. The Matthews Correlation Coefficient (MCC) provides a more reliable metric for binary classification because it achieves a high score only if the classifier performs well across all four quadrants of the confusion matrix.

Furthermore, model calibration is critical when automated systems provide probabilistic risk estimates for human review. Expected Calibration Error (ECE) and Brier scores provide quantitative measures of confidence alignment, while non-parametric bootstrap resampling enables estimation of sample variance and metric confidence intervals.

### E. Visual Explainability and Its Forensic Limitations
Gradient-weighted Class Activation Mapping (Grad-CAM) has become a standard approach for visualizing spatial regions contributing to predictions. However, critical work in explainable machine learning has highlighted substantial limitations of post-hoc saliency methods. Adebayo et al. demonstrated that several gradient-based saliency methods are insensitive to model parameter randomization and data labeling, behaving similarly to edge detectors. Furthermore, post-hoc visual explanations do not establish causal feature reliance and can provide a false sense of security regarding model reliability.

---

## III. Problem Formulation and Research Questions

### A. Mathematical Formulation
Let $\mathcal{X} \subset \mathbb{R}^{H \times W \times C}$ denote the input space of normalized RGB facial image crops with height $H=224$, width $W=224$, and channels $C=3$. In a binary forensic setting, each face sample $x \in \mathcal{X}$ is assigned a ground-truth label $y \in \mathcal{Y} = \{0, 1\}$, where:
$$y = \begin{cases} 0, & \text{if } x \text{ is authentic (REAL)}, \\ 1, & \text{if } x \text{ is synthetically manipulated (FAKE)}. \end{cases}$$

A forensic detection model $f_\theta: \mathcal{X} \to \mathbb{R}$ parameterized by weights $\theta$ produces an unconstrained scalar logit $z = f_\theta(x)$. The predicted posterior probability of manipulation is obtained via the standard logistic sigmoid activation:
$$\hat{p} = P(y = 1 \mid x; \theta) = \sigma(z) = \frac{1}{1 + e^{-z}}.$$

Binary decision assignment is governed by an operational threshold $\tau \in [0, 1]$:
$$\hat{y} = \begin{cases} 1, & \text{if } \hat{p} \ge \tau, \\ 0, & \text{if } \hat{p} < \tau. \end{cases}$$
Throughout this study, the operational threshold is strictly fixed at $\tau = 0.50$ across all evaluated models, without post-hoc threshold tuning on the test partition.

### B. Identity and Source Sequence Leakage Modeling
Consider an authentic video corpus $\mathcal{V} = \{v_1, v_2, \dots, v_M\}$ containing $M$ sequences of distinct individuals. A synthetic manipulation pipeline generates manipulated videos $v_{ij}^{\text{fake}}$ by transferring expressions or identities from source sequence $v_i$ onto target sequence $v_j$.

If video sequences are partitioned into training ($\mathcal{D}_{\text{train}}$), validation ($\mathcal{D}_{\text{val}}$), and test ($\mathcal{D}_{\text{test}}$) sets without tracking pairwise relationships, frames originating from $v_i$ or $v_j$ may reside in $\mathcal{D}_{\text{train}}$ while manipulated sequences $v_{ij}^{\text{fake}}$ are assigned to $\mathcal{D}_{\text{test}}$. In that event:
$$\text{Identity}(x_{\text{test}}) \in \text{Identities}(\mathcal{D}_{\text{train}}).$$
Because deep convolutional backbones contain millions of parameters, the network minimizes empirical risk by learning a high-dimensional facial identity embedding $\phi(x)$. At test time, the network recognizes the familiar facial morphology of Actor $i$ and predicts $y$ based on identity correlation rather than residual synthesis artifacts $\Delta x = x_{\text{fake}} - x_{\text{real}}$. Consequently, the model collapses when presented with novel, unseen individuals.

### C. Analytical Derivation of Class Imbalance Dynamics
In multi-method deepfake benchmarks, each authentic sequence undergoes manipulation by $K$ distinct synthesis techniques ($K \in \{4, 5, 6\}$). Consequently, the empirical dataset exhibits strong class imbalance, with positive prevalence:
$$\pi_1 = P(y = 1) = \frac{N_{\text{fake}}}{N_{\text{real}} + N_{\text{fake}}}.$$
In the FaceForensics++ C23 training partition, $N_{\text{real}} = 6,964$ and $N_{\text{fake}} = 43,742$, yielding an empirical training prevalence of $\pi_1 = 43,742 / 50,706 \approx 0.8626$ (a fake-to-real ratio of $6.28:1$).

Under standard empirical risk minimization with unweighted Binary Cross-Entropy (BCE) loss:
$$\mathcal{L}_{\text{BCE}}(z, y) = - \left[ y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right],$$
the gradient with respect to output logit $z$ is:
$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \sigma(z) - y.$$

Taking the expectation under the natural training data distribution where $\mathbb{E}[y] = \pi_1$, the expected gradient evaluated at a constant logit output evaluates to:
$$\mathbb{E}\left[\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z}\right] = \sigma(z) - \mathbb{E}[y] = \sigma(z) - \pi_1.$$

Setting the expected gradient to zero yields the stationary condition:
$$\sigma(z^*) = \pi_1 \implies z^* = \log\left(\frac{\pi_1}{1 - \pi_1}\right).$$
Because $\pi_1 \approx 0.8626 > 0.50$, the optimal constant logit satisfies $z^* \approx \log(6.28) \approx +1.837 > 0$. When thresholded at $\tau = 0.50$, this constant predictor outputs $\hat{y} = 1$ for all inputs. Evaluating this trivial majority predictor on the held-out test partition ($N_{\text{test}} = 9,589$, with $N_{\text{fake}} = 8,093$ and $N_{\text{real}} = 1,496$, where test prevalence is $\pi_{1,\text{test}} = 8,093 / 9,589 \approx 0.8440$) yields:
$$\text{Accuracy} = \frac{8,093 + 0}{9,589} = 84.40\%, \quad \text{Recall} = 1.0000, \quad \text{Specificity} = 0.0000, \quad \text{F1} = 0.9154, \quad \text{MCC} = 0.0000.$$

This derivation demonstrates that under severe prevalence skew, a model with zero diagnostic capability can exhibit high raw accuracy ($84.40\%$) and high F1 score ($0.9154$). Consequently, evaluating forensic models using accuracy alone can obscure severe failure on authentic media.

To balance the total loss contributions of authentic ($y=0$) and manipulated ($y=1$) samples across the training partition, the positive-class weight within PyTorch's `BCEWithLogitsLoss` is set to the inverse class ratio:
$$w_{\text{pos}} = \frac{N_{\text{real}}}{N_{\text{fake}}} = \frac{6,964}{43,742} \approx 0.159206.$$
Under this formulation, total loss penalties summed across all authentic and synthetic training instances are equalized:
$$N_{\text{fake}} \cdot w_{\text{pos}} = 43,742 \times \frac{6,964}{43,742} = 6,964 = N_{\text{real}} \cdot 1.0.$$

### D. Research Questions
This investigation is structured around six core research questions:
- **RQ1:** How does identity-disjoint partitioning affect evaluation reliability when compared to unverified splitting protocols?
- **RQ2:** How does severe class imbalance influence conventional evaluation metrics, and can positive-class weighting prevent majority-class collapse?
- **RQ3:** How do convolutional architectures (ResNet50, Xception, EfficientNet-B0, MobileNetV3-Large) compare to Vision Transformers (ViT-B/16) on compressed facial forensic media?
- **RQ4:** What performance improvements are associated with multi-stage regularization and validation-driven checkpoint selection on ResNet50?
- **RQ5:** Which manipulation categories and image perturbations present the greatest detection challenges?
- **RQ6:** What do probabilistic calibration and Grad-CAM spatial activation analyses reveal regarding model confidence and visual decision boundaries?

---

## IV. Dataset and Identity-Disjoint Experimental Protocol

### A. FaceForensics++ C23 Benchmark
Our experimental evaluation utilizes FaceForensics++ (FF++), comprising 1,000 pristine authentic video sequences manipulated via multiple algorithms across standardized H.264 compression tiers. All experiments are conducted on the lightly compressed H.264 (C23) tier (Constant Rate Factor 23), which closely approximates social media video distribution quality.

### B. Graph Connected-Components Partitioning
To formalize the sequence assignment problem, we model the benchmark video relationships as an undirected bipartite graph $G = (V, E)$. The vertex set $V = \{v_1, v_2, \dots, v_M\}$ represents all unique original video identifiers, and each manipulation pairing forms an undirected edge $e = (v_i, v_j) \in E$.

We implement a Disjoint-Set Union-Find algorithm with path compression and union by rank. Connected components are assigned atomically to training, validation, and held-out test partitions. This guarantees that:
$$\text{ID}(x_a) \cap \text{ID}(x_b) = \emptyset, \quad \forall x_a \in \mathcal{D}_{\text{train}}, \; x_b \in (\mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}).$$

### C. Leakage Verification and Reproducibility Audit
An initial data audit identified a partition contamination issue caused by:
1. An outdated splitting script on the local execution environment that failed to cluster transitive pairs across certain manipulation directories.
2. Incomplete filename parsing for DeepFakeDetection (DFD) video sequences containing double-underscore delimiters (`__`).

Once detected, the parsing routines were corrected to capture full compound identifier strings across all directories. The graph partitioning algorithm was re-executed and verified by checking the intersection of tracked video and actor identifiers across the resulting partitions:
$$\text{Identities}(\mathcal{D}_{\text{train}}) \cap \text{Identities}(\mathcal{D}_{\text{val}}) = \emptyset, \quad \text{Identities}(\mathcal{D}_{\text{train}}) \cap \text{Identities}(\mathcal{D}_{\text{test}}) = \emptyset, \quad \text{Identities}(\mathcal{D}_{\text{val}}) \cap \text{Identities}(\mathcal{D}_{\text{test}}) = \emptyset.$$
All downstream experiments were conducted strictly on these post-correction, frozen partitions.

### D. Face Extraction and Preprocessing
Frames are sampled at a uniform temporal stride of 10 frames. Facial bounding boxes are extracted using a deep convolutional face detector and expanded by a scale factor of $\alpha = 1.30$:
$$x_{\text{min}}' = x_c - 1.30 \frac{w}{2}, \quad y_{\text{min}}' = y_c - 1.30 \frac{h}{2}.$$
Expanding the crop margin by 30% ensures that critical boundary artifacts---including jawline blending contours, forehead Poisson interpolation seams, and perioral transitions---are preserved within the crop matrix. Extracted crops are resized via bilinear interpolation to $224 \times 224 \times 3$ and normalized using ImageNet channel parameters.

| Partition | Authentic (Real) | Manipulated (Fake) | Total | Ratio (F:R) |
|---|---|---|---|---|
| **Train** | 6,964 | 43,742 | 50,706 | 6.28 : 1 |
| **Validation** | 1,497 | 7,481 | 8,978 | 5.00 : 1 |
| **Held-Out Test** | 1,496 | 8,093 | 9,589 | 5.41 : 1 |
| **Total** | **9,957** | **59,316** | **69,273** | **5.96 : 1** |

---

## V. Model Architectures

We systematically evaluate six neural network architectures representing distinct design paradigms:
1. **Custom 4-Stage CNN Baseline:** Trained from scratch without ImageNet pre-training. Four convolutional blocks ($C=[32, 64, 128, 256]$) with BatchNorm, ReLU, and MaxPool, followed by adaptive average pooling and linear classification head. Parameter count: 93,825.
2. **ResNet50:** 50-layer deep residual network utilizing bottleneck building blocks. Pre-trained on ImageNet-1k, with classification head replaced by a single forensic logit. Parameter count: 23,510,081.
3. **Xception:** Factorizes standard convolutions into depthwise separable convolutions (Entry, Middle, Exit flows with residual shortcuts), decoupling spatial from cross-channel correlation. Parameter count: 20,809,001.
4. **EfficientNet-B0:** Utilizes mobile inverted bottleneck convolutions (MBConv) augmented with Squeeze-and-Excitation (SE) channel attention and compound scaling. Parameter count: 4,008,829.
5. **MobileNetV3-Large:** Combines hardware-aware NAS, depthwise separable convolutions, lightweight SE modules, and hard-swish activation functions. Parameter count: 1,518,881.
6. **Vision Transformer (ViT-B/16):** Partitions $224 \times 224$ images into 196 non-overlapping $16 \times 16$ patches, processed via 12 stacked transformer encoder layers with Multi-Head Self-Attention. Parameter count: 21,666,049.

| Architecture | Pretrained | Total Params | Trainable Params | Checkpoint Size |
|---|---|---|---|---|
| **Custom CNN** | None | 93,825 | 93,825 | 1.1 MB |
| **MobileNetV3-Large** | ImageNet-1k | 1,518,881 | 1,518,881 | 17.6 MB |
| **EfficientNet-B0** | ImageNet-1k | 4,008,829 | 4,008,829 | 46.4 MB |
| **Xception** | ImageNet-1k | 20,809,001 | 20,809,001 | 238.6 MB |
| **ViT-B/16** | ImageNet-1k | 21,666,049 | 21,666,049 | 248.1 MB |
| **ResNet50** | ImageNet-1k | 23,510,081 | 23,510,081 | 269.5 MB |

---

## VI. Training and Evaluation Methodology

### A. Loss Function and Optimization
Training optimizes Binary Cross-Entropy with Logits loss:
$$\mathcal{L}(z, y) = - \left[ w_{\text{pos}} \cdot y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right].$$
For class-balanced experiments, $w_{\text{pos}} = 0.159206$; for unweighted baselines, $w_{\text{pos}} = 1.0$.

Optimization is performed using AdamW with decoupled weight decay $\lambda = 10^{-4}$, initial learning rate $\eta_0 = 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and cosine annealing schedule with a 2-epoch linear warmup.

**Batch Size Configuration:** Reconciled from experimental artifacts, the verified ResNet50 Regularized run utilized a mini-batch size of **64**, whereas baseline comparison runs operated at a mini-batch size of **32**. All models were trained for up to 20 epochs with random seed $S=42$.

### B. Multi-Stage Regularization Suite
The regularized configuration incorporates:
- **Spatial Data Augmentation:** Random horizontal reflections ($p=0.50$), random affine rotations ($\theta \in [-15^\circ, +15^\circ]$), brightness jitter ($\pm 0.20$), contrast jitter ($\pm 0.20$), and color jitter (hue $\pm 0.05$, saturation $\pm 0.20$). Notably, Gaussian blurring and JPEG compression were omitted from the training pipeline to avoid destroying high-frequency forensic clues.
- **Dropout Regularization:** Dropout rate $p = 0.50$ inserted prior to the final projection layer.
- **Validation ROC-AUC Checkpoint Selection:** Model selection was governed strictly by peak validation ROC-AUC ($\text{val\_roc\_auc} \to \max$) with early stopping patience of 5 epochs ($\delta = 0.001$). Held-out test data was not used for checkpoint selection.

### C. Evaluation Metrics and Non-Parametric Bootstrap Protocol
Evaluation employs: Accuracy, Recall (Sensitivity), Specificity, Precision, F1 Score, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Cohen's Kappa ($\kappa$), ROC-AUC, PR-AUC, Expected Calibration Error (ECE), and Brier Score.

To quantify finite-sample variance, we perform non-parametric bootstrap resampling with $B = 1,000$ iterations (seed $S=42$, test size $N=9,589$) using the percentile method.

**Methodological Clarification:** Bootstrap confidence intervals quantify sample uncertainty around observed estimates. However, because no formal paired hypothesis test was conducted, these intervals should not be interpreted as a standalone test of statistical significance.

---

## VII. Experimental Results

### A. Master Multi-Architecture Benchmark
Evaluation results across the 9,589 frozen test images ($\tau = 0.50$) are summarized below:

| Architecture | Accuracy | Precision | Recall | Specificity | F1 Score | ROC-AUC | PR-AUC | Bal. Acc. | MCC | Cohen's $\kappa$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **ResNet50 Regularized** | **0.9389** | 0.9755 | **0.9514** | 0.8710 | **0.9633** | **0.9718** | **0.9938** | 0.9112 | **0.7821** | **0.7820** |
| **Xception** | 0.9370 | **0.9777** | 0.9470 | **0.8830** | 0.9621 | 0.9706 | 0.9931 | **0.9150** | 0.7798 | 0.7795 |
| **EfficientNet-B0** | 0.9208 | 0.9721 | 0.9330 | 0.8549 | 0.9521 | 0.9547 | 0.9879 | 0.8940 | 0.7290 | 0.7288 |
| **ViT-B/16** | 0.9005 | 0.9663 | 0.9140 | 0.8275 | 0.9394 | 0.9512 | 0.9894 | 0.8708 | 0.6706 | 0.6701 |
| **MobileNetV3-Large** | 0.8797 | 0.9444 | 0.9110 | 0.7099 | 0.9274 | 0.9187 | 0.9820 | 0.8105 | 0.5792 | 0.5786 |
| *Ablation Variants:* | | | | | | | | | | |
| Custom CNN (Balanced) | 0.4324 | 0.8895 | 0.3739 | 0.7487 | 0.5265 | 0.6196 | 0.8966 | 0.5613 | 0.0987 | 0.0841 |
| Custom CNN (Unweighted) | 0.8440 | 0.8440 | 1.0000 | 0.0000 | 0.9154 | 0.5607 | 0.8799 | 0.5000 | 0.0000 | 0.0000 |
| ResNet50 Baseline | 0.9133 | 0.9709 | 0.9250 | 0.8503 | 0.9474 | 0.9578 | 0.9914 | 0.8876 | 0.7087 | 0.7083 |

**Key Findings:**
1. **ResNet50 Regularized** achieved top performance across primary discriminative metrics ($\text{Accuracy} = 0.9389$, $\text{F1} = 0.9633$, $\text{ROC-AUC} = 0.9718$, $\text{PR-AUC} = 0.9938$, $\text{MCC} = 0.7821$).
2. **Xception** demonstrated comparable performance ($\text{F1} = 0.9621$, $\text{ROC-AUC} = 0.9706$), achieving the highest observed precision ($0.9777$) and specificity ($0.8830$).
3. **Convolutional vs. Vision Transformer Biases:** Convolutional architectures outperformed ViT-B/16 ($\text{Accuracy} = 0.9005$, $\text{F1} = 0.9394$, $\text{ROC-AUC} = 0.9512$). One plausible hypothesis is that localized manipulation artifacts benefit from the spatial locality and translation-equivariant inductive biases of convolutional filters, whereas patch tokenization may segment subtle edge discontinuities.
4. **MobileNetV3-Large** attained an F1 score of $0.9274$ and ROC-AUC of $0.9187$, but exhibited lower specificity ($0.7099$), reflecting channel reduction in early layers.

### B. Controlled Class-Imbalance Ablation
- **Unweighted Baseline:** Collapsed completely to majority-class prediction: $\text{Recall} = 1.0000$, $\text{Specificity} = 0.0000$, $\text{MCC} = 0.0000$, classifying all 1,496 authentic test images as manipulated. Yet, its accuracy was $84.40\%$ and F1 was $0.9154$.
- **Class-Balanced CNN:** Incorporating $w_{\text{pos}} = 0.1592$ restored decision boundaries, lifting specificity from $0.00\%$ to $74.87\%$, ROC-AUC from $0.5607$ to $0.6196$, and MCC from $0.0000$ to $0.0987$. While overall accuracy fell to $43.24\%$ due to the low capacity of a scratch-trained CNN, class weighting prevented majority-class collapse.

### C. Regularization Ablation
Comparing ResNet50 Baseline against ResNet50 Regularized:
- Accuracy increased by $+2.56\%$ ($0.9133 \to 0.9389$).
- Specificity increased by $+2.07\%$ ($0.8503 \to 0.8710$).
- ROC-AUC improved by $+1.40\%$ ($0.9578 \to 0.9718$).
- MCC improved by $+0.0734$ ($0.7087 \to 0.7821$).

### D. Bootstrap Uncertainty Estimation ($B = 1,000$)
- **ResNet50 Regularized:** Accuracy $[0.9339, 0.9436]$ (mean $0.9389 \pm 0.0025$); F1 Score $[0.9603, 0.9663]$ (mean $0.9634 \pm 0.0015$); ROC-AUC $[0.9676, 0.9759]$ (mean $0.9719 \pm 0.0021$).
- **Xception:** Accuracy $[0.9322, 0.9422]$ (mean $0.9371 \pm 0.0025$); F1 Score $[0.9591, 0.9653]$ (mean $0.9621 \pm 0.0015$); ROC-AUC $[0.9662, 0.9747]$ (mean $0.9707 \pm 0.0021$).
The overlapping confidence intervals reflect comparable empirical performance across top models.

---

## VIII. Error, Calibration, and Robustness Analysis

### A. Sub-Category Manipulation-Wise Breakdown
| Manipulation Category | True Class | Samples ($N$) | Recall / Spec. | Mean Fake Prob. | Errors | Error Share (\%) |
|---|---|---|---|---|---|---|
| **DeepFakeDetection** | FAKE | 615 | **99.67\%** | 0.9889 | $\text{FN} = 2$ | 0.5\% |
| **FaceSwap** | FAKE | 1,496 | 97.33\% | 0.9658 | $\text{FN} = 40$ | 10.2\% |
| **Deepfakes** | FAKE | 1,492 | 97.32\% | 0.9657 | $\text{FN} = 40$ | 10.2\% |
| **Face2Face** | FAKE | 1,499 | 97.00\% | 0.9622 | $\text{FN} = 45$ | 11.5\% |
| **FaceShifter** | FAKE | 1,495 | 96.86\% | 0.9568 | $\text{FN} = 47$ | 12.0\% |
| **NeuralTextures** | FAKE | 1,496 | **85.36\%** | 0.8363 | $\text{FN} = 219$ | **55.7\%** |
| **Original (Authentic)** | REAL | 1,496 | **87.10\%** | 0.1497 | $\text{FP} = 193$ | --- |

**The NeuralTextures Vulnerability:** Out of 393 total false negatives, **219 (55.7\%)** originated from NeuralTextures alone. Unlike full-face replacement pipelines that leave visible boundary seams across the jawline and forehead, NeuralTextures modifies only the mouth cavity and perioral region. The remaining 90\% of the facial area is authentic, allowing these subtle modifications to evade global pooling operations.

### B. Confusion Matrix and Error Audit
Out of 9,589 test samples, ResNet50 Regularized produced 586 classification errors:
- $\text{True Positives (TP)} = 7,700$
- $\text{True Negatives (TN)} = 1,303$
- $\text{False Positives (FP)} = 193$ ($\text{FPR} = 12.90\%$)
- $\text{False Negatives (FN)} = 393$ ($\text{FNR} = 4.86\%$)

**Confidence Band Audit:**
- False Positives ($N=193$): Mean predicted fake probability was $\hat{p} = 0.8501$, with 125 samples exhibiting $\hat{p} > 0.80$, and 104 samples residing in $[0.90, 1.00]$. Motion blur and compression blockiness contributed to these false alarms.
- False Negatives ($N=393$): Mean predicted probability was $\hat{p} = 0.1686$, with 249 samples exhibiting $\hat{p} < 0.20$, and 182 samples falling in $[0.00, 0.10]$ (predominantly NeuralTextures).
- The uncertain mid-range interval $[0.30, 0.70]$ accounted for 138 of 586 total errors (23.5\%).

### C. Probabilistic Calibration
- $\text{ECE} = \mathbf{0.0418}$ ($4.18\%$), $\text{Brier Score} = \mathbf{0.0524}$.
- Over 80\% of test samples ($7,736 / 9,589$) reside in extreme bins ($[0.0, 0.1]$ and $[0.9, 1.0]$), with calibration gaps of $0.0027$ and $0.0074$.

### D. Perturbation Robustness Stress-Testing ($N=502$ Synthetic Subset)
| Perturbation Condition | $N$ | Recall (\%) | Mean Fake Prob. | Retention (\%) |
|---|---|---|---|---|
| **Clean Baseline** | 502 | **98.21\%** | 0.9699 | 100.0\% |
| **JPEG Quality 90** | 502 | 94.82\% | 0.9289 | 96.5\% |
| **JPEG Quality 70** | 502 | 96.41\% | 0.9384 | 98.2\% |
| **JPEG Quality 50** | 502 | 72.51\% | 0.7135 | 73.8\% |
| **JPEG Quality 30** | 502 | **39.64\%** | 0.4029 | **40.4\%** |
| **Gaussian Blur ($r = 1.5$)** | 502 | 100.00\% | 0.9992 | 101.8\% |
| **Gaussian Noise ($\sigma = 0.05$)** | 502 | 99.40\% | 0.9685 | 101.2\% |
| **Downscale 4x + Upscale** | 502 | 100.00\% | 0.9955 | 101.8\% |

Performance remained relatively stable under moderate compression ($98.2\%$ retention at JPEG Q=70), but degraded sharply under stronger quantization ($39.64\%$ recall at Q=30), consistent with the loss of fine spatial gradients supporting forensic discrimination.

---

## IX. Visual Explainability via Grad-CAM

Grad-CAM was computed at the final residual block of ResNet50 (`model.backbone.layer4[-1]`):
- **Boundary Localization:** Saliency heatmaps for face-swapping manipulations (Deepfakes, FaceSwap) concentrate along the jawline, chin contours, and facial boundaries where blending algorithms merge skin textures.
- **Perioral Focus:** Reenactment manipulations (Face2Face, NeuralTextures) show activations focused on the lips, nasolabial folds, and mouth interior.
- **Background Suppression:** Activations across background regions and clothing remain low across examined samples.

**Forensic Limitations:** While Grad-CAM confirms spatial activation alignment with anatomical boundaries, post-hoc saliency methods do not establish causal reliance or prove the absence of shortcut learning. Saliency maps provide qualitative diagnostic visualization but cannot verify model generalizability under out-of-distribution conditions.

---

## X. Computational Complexity and Efficiency

| Architecture | Parameters | Checkpoint Size | CPU Latency | Throughput |
|---|---|---|---|---|
| **Custom CNN** | 93,825 | 1.1 MB | 17.0 $\pm$ 0.8 ms | 58.9 FPS |
| **MobileNetV3-Large** | 1,518,881 | 17.6 MB | 28.2 $\pm$ 3.1 ms | 35.4 FPS |
| **EfficientNet-B0** | 4,008,829 | 46.4 MB | 64.0 $\pm$ 5.3 ms | 15.6 FPS |
| **ViT-B/16** | 21,666,049 | 248.1 MB | 110.0 $\pm$ 22.9 ms | 9.1 FPS |
| **ResNet50** | 23,510,081 | 269.5 MB | 128.9 $\pm$ 9.0 ms | 7.8 FPS |
| **Xception** | 20,809,001 | 238.6 MB | 167.4 $\pm$ 22.6 ms | 6.0 FPS |

MobileNetV3-Large provides the lowest latency among pre-trained backbones (28.2 ms, 35.4 FPS) with a 17.6 MB checkpoint footprint, while ResNet50 and Xception provide higher forensic discriminative performance (F1 = 0.9633 and 0.9621) at the cost of higher parameter counts and latency.

---

## XI. Discussion

1. **Identity-Disjoint Splitting:** Atomically assigning connected components ensures reported metrics reflect generalizable manipulation detection rather than biometric recognition.
2. **Pathology of Conventional Accuracy:** When the positive class comprises over 84% of samples, an unweighted classifier can collapse to majority prediction while achieving apparent 84.40% accuracy. Specificity, balanced accuracy, and MCC provide more reliable assessments.
3. **Inductive Biases:** Convolutional architectures outperformed ViT-B/16 on compressed $224 \times 224$ images, consistent with the utility of translation equivariance for localized boundary artifacts.
4. **NeuralTextures Challenge:** Accounted for 55.7% of false negatives, showing that localized mouth modifications can evade global spatial pooling.
5. **Compression Sensitivity:** Sharp degradation under JPEG Q=30 highlights vulnerability to aggressive quantization.
6. **Calibration Quality:** While overall ECE was 4.18%, confident false alarms indicate that calibration should be carefully evaluated in decision-support contexts.
7. **Explainability Bounds:** Grad-CAM provides qualitative diagnostic visualization but does not establish causal feature reliance.

---

## XII. Threats to Validity and Limitations

1. **Single Benchmark Constraint:** All experiments were conducted on FaceForensics++ C23; cross-dataset evaluations (Celeb-DF, DFDC) were not conducted.
2. **Absence of Temporal Modeling:** Evaluated models operate strictly on individual static frames.
3. **Perturbation Subset Scale:** Perturbation stress tests were evaluated on an $N = 502$ synthetic image subset.
4. **Absence of Adaptive Adversarial Attacks:** Targeted gradient-based adversarial attacks were not evaluated.
5. **Benchmark-Specific Manipulation Distribution:** Performance on emerging commercial or diffusion tools remains unmeasured.
6. **Qualitative Explainability:** Grad-CAM is a post-hoc diagnostic tool and does not provide formal guarantees regarding model feature reliance.
7. **Statistical Uncertainty vs. Hypothesis Testing:** Bootstrap confidence intervals quantify sample variance but do not substitute for formal paired hypothesis testing.

---

## XIII. Conclusion and Future Work

This study investigated deepfake detection on the compressed FaceForensics++ C23 benchmark using a verified identity-disjoint partitioning protocol. We demonstrated that unweighted loss functions can induce majority-class collapse under prevalence skew, producing deceptively high accuracy while failing completely on authentic media.

Among six evaluated architectures, ResNet50 Regularized and Xception demonstrated leading performance (F1 = 0.9633 and 0.9621). Sub-category failure audits revealed that NeuralTextures accounted for 55.7\% of all false negatives, highlighting localized oral reenactment as a primary challenge for global classifiers. Future work will focus on cross-dataset generalization, spatio-temporal video modeling, localized perioral architectures, and formal paired hypothesis testing.
