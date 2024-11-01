# CHANGELOG: Deepfake Detection Research Paper Transformation

This document outlines all major structural, methodological, numerical, formatting, and editorial revisions executed to transform the draft manuscript into a submission-ready IEEE conference research paper.

---

### 1. IEEE Conference Format and Document Class Compliance
- **Replaced Document Class:** Changed from `\documentclass[journal,compsoc]{IEEEtran}` to `\documentclass[conference]{IEEEtran}`.
- **Removed Fake Journal Metadata:** Stripped all counterfeit volume/issue numbers, fictitious publication dates, and running headers (`\markboth{IEEE Transactions on Information Forensics and Security...}{...}`).
- **Removed Journal Title Blocks:** Eliminated `\IEEEtitleabstractindextext`, `\IEEEdisplaynontitleabstractindextext`, and `\IEEEpeerreviewmaketitle`, restoring standard IEEE conference title and abstract layout.
- **Clean Author Information:** Confirmed author as **Saket Mathur** with clean, minimal affiliation (*Department of Computer Science and Engineering*). Removed all invented email addresses, fake institutional URLs, and placeholder links.

### 2. Removal of Project Branding and Marketing Identity
- **Complete Elimination of Named System:** Removed all occurrences of "DeepShield-X", "DeepShield", and system-level branding from the title, abstract, body text, figures, tables, and acknowledgment.
- **Adoption of Neutral Academic Language:** Replaced promotional phrasing ("We introduce DeepShield-X...") with standard scientific constructions ("In this study, we present a systematic experimental investigation...", "the proposed evaluation protocol", "the investigated architectures").
- **Academic Title:** Renamed manuscript to:
  > *"Robust and Explainable DeepFake Detection Using Identity-Disjoint Evaluation of Convolutional Neural Networks and Vision Transformers"*

### 3. Elimination of Markdown Remnants from LaTeX Source
- **Zero Markdown Formatting:** Eradicated all `**bold**`, `__italic__`, and Markdown header artifacts previously embedded in the `.tex` source.
- **Proper LaTeX Syntax:** Verified that all text formatting utilizes valid LaTeX macros (`\textbf{...}`, `\textit{...}`, `\texttt{...}`).

### 4. Mathematical Derivations and Class Imbalance Rectification
- **Rigorous Gradient Derivation:** Corrected the gradient derivation of unweighted Binary Cross-Entropy (BCE) loss. Replaced the flawed "balanced mini-batch" assertion with an exact analytical derivation under training prevalence skew ($\pi_1 = 43,742 / 50,706 \approx 0.8626$). Demonstrated that setting the expected gradient to zero produces a constant positive logit ($z^* \approx +1.837$), explaining why unweighted models collapse toward all-fake majority prediction.
- **PyTorch BCE Loss Weighting Derivation:** Formulated the exact positive-class weighting in PyTorch's `BCEWithLogitsLoss`:
  $$w_{\text{pos}} = \frac{N_{\text{real}}}{N_{\text{fake}}} = \frac{6,964}{43,742} \approx 0.159206$$
  Proved that this weighting equalizes total gradient penalties ($43,742 \times 0.159206 = 6,964$) between authentic and synthetic classes across an epoch.

### 5. Numerical Reconciliations Against Raw Experimental Artifacts
- **Dataset Partition Counts:** Verified exact FaceForensics++ C23 extracted face crop numbers:
  - Train: 50,706 ($6,964$ Real, $43,742$ Fake; $6.28:1$)
  - Validation: 8,978 ($1,497$ Real, $7,481$ Fake; $5.00:1$)
  - Test: 9,589 ($1,496$ Real, $8,093$ Fake; $5.41:1$)
  - Total: 69,273 ($9,957$ Real, $59,316$ Fake; $5.96:1$)
- **ResNet50 Regularized Confusion Matrix:** Confirmed exact counts ($N=9,589$):
  - True Positives (TP) = $7,700$
  - True Negatives (TN) = $1,303$
  - False Positives (FP) = $193$
  - False Negatives (FN) = $393$
  - Total Errors = $586$ ($\text{FPR} = 12.90\%$, $\text{FNR} = 4.86\%$)
- **Batch Size Reconciliation:** Resolved discrepancies by documenting the actual evaluated training configurations: batch size **64** for ResNet50 Regularized and batch size **32** for architecture comparison baselines.
- **Confidence Band Re-audit:** Removed conflicting and unsupported assertions ("only 34 samples incorrectly classified with $p > 0.85$" and "over 94% of errors occur in $[0.35, 0.65]$"). Replaced with exact counts from `error_analysis_summary.json`:
  - 125 false positives have $\hat{p} > 0.80$ (104 in $[0.90, 1.00]$).
  - 249 false negatives have $\hat{p} < 0.20$ (182 in $[0.00, 0.10]$).
  - 138 errors fall within the ambiguous interval $[0.30, 0.70]$ ($23.5\%$ of errors).
- **Sub-Category Manipulation Breakdown:** Verified exact error counts across all six manipulations:
  - DeepFakeDetection: $N=615$, Recall $99.67\%$, $\text{FN} = 2$
  - FaceSwap: $N=1,496$, Recall $97.33\%$, $\text{FN} = 40$
  - Deepfakes: $N=1,492$, Recall $97.32\%$, $\text{FN} = 40$
  - Face2Face: $N=1,499$, Recall $97.00\%$, $\text{FN} = 45$
  - FaceShifter: $N=1,495$, Recall $96.86\%$, $\text{FN} = 47$
  - NeuralTextures: $N=1,496$, Recall $85.36\%$, $\text{FN} = 219$ ($55.7\%$ of total false negatives)
  - Authentic Original: $N=1,496$, Specificity $87.10\%$, $\text{FP} = 193$
- **Robustness Scope Disclosure:** Explicitly bounded perturbation stress-testing to the evaluated $N=502$ synthetic image subset, reporting exact retention figures (Q90: 96.5\%, Q70: 98.2\%, Q50: 73.8\%, Q30: 40.4\%).

### 6. Scientific Claim Control and Defensibility
- **Elimination of Hyperbolic Claims:** Removed all unverified claims of "universal robustness", "provably leak-free", "irrefutable proof", "guaranteed", "state-of-the-art", and "immunized models".
- **Grounded Interpretation of Inductive Biases:** Framed the superior performance of CNNs over Vision Transformers in this setting as an empirical finding consistent with the spatial locality and translation equivariance of convolutional kernels on compressed boundary artifacts, rather than an absolute theoretical proof.
- **Grad-CAM Epistemology:** Explicitly acknowledged that Grad-CAM is a post-hoc qualitative localization technique that cannot establish causal reliance, guarantee generalizability, or rule out subtle shortcut learning.
- **Statistical Significance Nuance:** Bounded bootstrap confidence intervals ($B=1,000$) as measures of finite-sample estimation variance rather than standalone paired hypothesis tests of significance.
- **Complexity Framing:** Renamed efficiency section to *"Computational Complexity and Efficiency"*, removing all deployment, production, edge-deployment, and real-time guarantees.

### 7. Bibliography and Literature Review Expansion
- **Verified References & Crossref Audit:** Conducted a comprehensive automated API audit of all 44 references using Crossref. 100% of DOIs resolve to their exact matching peer-reviewed works.
- **Resolution of Discrepancies:**
  - `verdoliva2020media`: Fixed journal to *IEEE Journal of Selected Topics in Signal Processing* (vol. 14, no. 5, pp. 910--932, 2020), corrected title to *"Media Forensics and {DeepFakes}: An Overview"*, and corrected DOI (`10.1109/JSTSP.2020.3002101`).
  - `durall2020watch`: Corrected title to official CVPR publication title: *"Watch Your Up-Convolution: {CNN} Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions"* and exact pagination (`7887--7896`).
  - `qian2020f3net`: Corrected ECCV LNCS DOI (`10.1007/978-3-030-58610-2_6`).
  - `haliassos2021lips`: Corrected CVPR DOI (`10.1109/CVPR46437.2021.00500`) and pagination (`5037--5047`).
  - `gu2021spatiotemporal`: Corrected ACM MM DOI (`10.1145/3474085.3475508`).
  - `lyu2020deepfake`: Corrected ICMEW DOI (`10.1109/ICMEW46912.2020.9105991`).
  - `zi2020wilddeepfake`: Corrected ACM MM DOI (`10.1145/3394171.3413769`).
  - `li2020faceshifter`: Corrected entry type to arXiv preprint (`arXiv:1912.13457`, 2019) reflecting archival status.
  - `nirkin2018faceswap`: Replaced FSGAN with Nirkin et al.'s landmark FG 2018 paper (*"On Face Segmentation, Face Swapping, and Face Perception"*, FG 2018, DOI: `10.1109/FG.2018.00024`), matching the 3D landmark, Delaunay triangulation, and Poisson blending method cited.
  - `dong2023implicit`: Added Shichao Dong et al., *"Implicit Identity Leakage: The Stumbling Block to Improving {DeepFake} Detection Generalization"*, CVPR 2023, DOI: `10.1109/CVPR52729.2023.00389`.
  - `geirhos2020shortcut`: Added Robert Geirhos et al., *"Shortcut Learning in Deep Neural Networks"*, *Nature Machine Intelligence*, 2020, DOI: `10.1038/s42256-020-00257-z`.
- **Comprehensive Academic Scope:** All 44 citations resolve to genuine, verified peer-reviewed papers or canonical preprints; zero unused, zero missing, and zero fabricated citations.

### 8. LaTeX Layout, Float Queueing, and Table Collision Rectification
- **Table IX & X Horizontal Overlap Fixed:**
  - *Root Cause:* Both Table IX (`tab:robustness`) and Table X (`tab:efficiency`) were single-column tables with 5 wide columns exceeding the IEEEtran column width ($\sim 3.5\text{ in} = 252\text{ pt}$). When placed at the top of column 1 and column 2 on the same page, the overflowing right margin of Table IX smashed directly into the overflowing left margin of Table X.
  - *Fix:* Wrapped all single-column tables (Tables I, II, IV, V, VI, VIII, IX, X) in `\resizebox{\columnwidth}{!}{% \begin{tabular}... \end{tabular}%}` and streamlined column headers (e.g., `Mean Fake $P$`, `4$\times$`, `Params`, `Size`, `CPU Latency`). Guarantees zero horizontal overflow into adjacent columns.
- **Grad-CAM Oversized Float and Trailing Figures Fixed:**
  - *Root Cause:* The original `gradcam_sample_grid.png` was $2906 \times 5792$ pixels (aspect ratio $0.50$). At `width=0.92\textwidth`, its physical height was $13.16\text{ inches}$, exceeding the standard page `\textheight` ($9.25\text{ inches}$). Because LaTeX cannot place a float taller than a page, it queued Figure 9 indefinitely, which in turn blocked Figure 10 (FIFO queue), deferring both figures all the way to `\end{document}` *after* the bibliography and leaving large empty page gaps.
  - *Fix:* Re-composed the Grad-CAM visualizations into a $2 \times 2$ grid (`gradcam_sample_grid.png`, dimensions $5852 \times 1930$, aspect ratio $3.03$). Constrained height to `2.35in` with `keepaspectratio`. Sits neatly as a two-column top float occupying $< 25\%$ of the page height.
- **Figure 10 & 8 Sizing Constrained:** Added explicit `height` bounds and `keepaspectratio` to Figure 8 ($2.0\text{ in}$), Figure 7 ($2.3\text{ in}$), and Figure 10 ($2.2\text{ in}$), allowing LaTeX to integrate single-column floats harmoniously alongside text.
- **Float Placement & Barrier Protection:**
  - Added `\usepackage{placeins}` and placed a definitive `\FloatBarrier` immediately before `\section*{Acknowledgment}` and `\bibliographystyle{IEEEtran}`, mathematically guaranteeing that zero floats can spill past the conclusion or after the references.
  - Optimized IEEE float allocation fractions (`\topfraction=0.95`, `\bottomfraction=0.85`, `\textfraction=0.05`, `\floatpagefraction=0.85`, `\dbltopfraction=0.95`, `\dblfloatpagefraction=0.85`) to avoid premature float deferrals.

