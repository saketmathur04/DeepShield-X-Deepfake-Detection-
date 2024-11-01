# FINAL SUBMISSION CHECKLIST: IEEE CONFERENCE MANUSCRIPT

**Manuscript Title:** *Robust and Explainable DeepFake Detection Using Identity-Disjoint Evaluation of Convolutional Neural Networks and Vision Transformers*  
**Author:** Saket Mathur  
**Target Venue:** IEEE Conference (Computer Vision / Multimedia Forensics)  
**Main Source File:** `deepshield_x_paper.tex`  
**Bibliography File:** `references.bib`  

---

### Critical Editorial & Methodological Criteria

- [x] **No Project / Product Name:** Zero occurrences of "DeepShield-X", "DeepShield", or product branding in title, body, tables, or figures.
- [x] **Correct IEEE Conference LaTeX Class:** Uses `\documentclass[conference]{IEEEtran}`.
- [x] **No Fake Journal Metadata:** Stripped `\markboth`, fake DOIs, fake volume/issue, and fake dates.
- [x] **Truthful Author Information:** Author listed as **Saket Mathur** with clean minimal affiliation (*Department of Computer Science and Engineering*). No fabricated emails, ORCIDs, or URLs.
- [x] **No Markdown Remnants:** Zero occurrences of `**`, `__`, or Markdown formatting inside LaTeX source.
- [x] **Target Page Length Achieved:** Manuscript structured into 13 substantive sections, with comprehensive Related Work, mathematical formulations, 10 detailed tables, and 10 figure environments naturally spanning approximately 13–14 IEEE two-column pages without artificial padding.
- [x] **Abstract Bounded:** Abstract is 238 words, strictly within the IEEE 200–250 word recommendation, containing problem, method, dataset, models, ablation, and key findings.
- [x] **Formal Keywords:** 8 precise IEEE forensic/vision keywords included.
- [x] **Substantive Related Work:** Covers deepfake generation, CNN/ViT detection architectures, dataset leakage, evaluation metrics, and explainability with 44 landmark references.
- [x] **Clear Research Questions:** Explicit formulation of RQ1 through RQ6 in Section III, with answers provided systematically in Section VII and Section XI.
- [x] **Identity-Disjoint Splitting Explained:** Graph formulation with bipartite edges, Disjoint-Set Union-Find (Algorithm 1) with path compression and union by rank.
- [x] **Leakage Verification Documented:** Tracked identifiers verified; initial local script and DFD delimiter leak transparently explained as a reproducibility finding.
- [x] **Dataset Numbers Verified:** Train ($50,706$), Validation ($8,978$), Test ($9,589$), Total ($69,273$) face crops verified across all tables.
- [x] **Class Imbalance Mathematics Corrected:** Exact expected gradient derivation under training prevalence skew ($\pi_1 \approx 0.8626$); exact PyTorch `pos_weight = 6,964 / 43,742 \approx 0.1592` loss formulation derived.
- [x] **Six Architectures Accurately Described:** Custom CNN (93,825), MobileNetV3-L (1.5M), EfficientNet-B0 (4.0M), Xception (20.8M), ViT-B/16 (21.7M), ResNet50 (23.5M).
- [x] **Actual Training Configuration Verified:** Documented batch size 64 for ResNet50 Regularized, batch size 32 for baselines, lr $10^{-4}$, AdamW, cosine schedule, warmup, seed 42.
- [x] **Validation-Driven Checkpoint Selection:** Monitored validation ROC-AUC (patience 5, delta 0.001); zero test-set threshold or checkpoint tuning.
- [x] **Master Benchmark Verified:** ResNet50 Reg. (F1 = 0.9633), Xception (F1 = 0.9621), EfficientNet-B0 (F1 = 0.9521), ViT-B/16 (F1 = 0.9394), MobileNetV3-L (F1 = 0.9274).
- [x] **Class Imbalance Ablation Retained:** Unweighted Custom CNN collapse (Acc 84.40%, Spec 0.00%, MCC 0.00) vs Balanced Custom CNN (Acc 43.24%, Spec 74.87%, MCC 0.0987).
- [x] **Regularization Ablation Retained:** Baseline vs Regularized ResNet50 comparison across all 10 metrics.
- [x] **Manipulation-Wise Breakdown Detailed:** Exact counts for all 6 manipulations + authentic original; NeuralTextures identified as 55.7% (219/393) of false negatives.
- [x] **Error & Confidence Audit Verified:** Replaced inaccurate "34 errors" statement with exact verified distribution: 125 FP with $p > 0.80$, 249 FN with $p < 0.20$, and 138 errors in ambiguous $[0.30, 0.70]$ band.
- [x] **Probabilistic Calibration Included:** ECE = 0.0418, Brier = 0.0524, 10-bin reliability analysis.
- [x] **Robustness Scope Disclosed:** Evaluated on $N=502$ synthetic image subset; exact retention figures reported across JPEG sweeps and spatial filters.
- [x] **Grad-CAM Claims Scientifically Limited:** Acknowledged as qualitative post-hoc localization that does not establish causal feature reliance or rule out shortcut bias.
- [x] **Bootstrap Claims Scientifically Limited:** Percentile CIs ($B=1,000$) quantify sample uncertainty; explicitly states no formal paired hypothesis test was conducted.
- [x] **Computational Complexity Profiling:** Measured on standard workstation CPU; latency and throughput reported without deployment, production, or real-time claims.
- [x] **Dedicated Discussion Section:** Synthesizes findings across 10 academic themes without repeating results.
- [x] **Threats to Validity Included:** All 9 threats to validity explicitly documented.
- [x] **Future Work Grounded:** Concrete future research directions outlined without claiming unperformed implementations.
- [x] **No Fabricated Experiments or Values:** Every single number matches project JSON/CSV artifacts.
- [x] **Complete Bibliography Resolution:** All 44 citations resolve to genuine, verified peer-reviewed papers; 0 missing, 0 unused.
- [x] **All Figures Verified:** 10 of 10 figures exist on disk in `Research/paper/images/` and match caption descriptions.
- [x] **All Cross-References Resolved:** 34 labels, 32 refs, 0 unresolved.
- [x] **LaTeX Syntax Clean:** Valid IEEE LaTeX compiling cleanly without errors.
- [x] **No Unsupported Causal Claims:** Replaced speculative claims with evidence-bounded interpretations.
- [x] **Original Academic Prose:** Derives all arguments from experimental evidence without plagiarism or detector evasion tricks.

---
**Verdict:** **APPROVED FOR SUBMISSION.**
