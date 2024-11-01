# DeepShield-X: Reproducibility Runbook

This document details the exact, reproducible workflow to audit, evaluate, and benchmark the DeepShield-X deepfake detection pipeline on Windows and Linux workstations.

---

## 1. Environment Setup

DeepShield-X runs in a standard Python 3.10+ virtual environment or Anaconda environment.

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepShield-X.git
cd DeepShield-X

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install pytorch-grad-cam
```

### Validate Installation & Integrity
```bash
# Verify all modules compile cleanly
python -m compileall src scripts tests

# Execute the complete automated test suite (40 unit and integration tests)
pytest -q
```
*Expected Result:* `40 passed` (0 errors, 0 failures).

---

## 2. Dataset Structure & Leakage Prevention

The project operates on the **FaceForensics++ C23** benchmark. To guarantee **zero identity leakage**, data splitting uses a graph connected-components (Union-Find) algorithm that isolates video pairs and actors prior to face extraction.

### Directory Hierarchy:
```text
data/
├── faces/
│   └── test/
│       └── fake/
│           └── Deepfakes/        # Extracted 224x224 face crops
└── splits/
    ├── train.csv                 # 50,706 samples (6,964 Real, 43,742 Fake)
    ├── val.csv                   # 8,978 samples (1,497 Real, 7,481 Fake)
    └── test.csv                  # 9,589 samples (1,496 Real, 8,093 Fake)
```

To re-generate splits from raw video annotations:
```bash
python scripts/create_splits.py
```

---

## 3. Checkpoint Inventory

Pre-trained and fine-tuned checkpoints are preserved under `experiments/checkpoints/`:
- `experiments/checkpoints/resnet50_regularized/resnet50_best.pth` (282.6 MB) — **Top Model**
- `experiments/checkpoints/resnet50/resnet50_best.pth` (282.6 MB) — Baseline ResNet
- `experiments/checkpoints/xception/xception_best.pth` (250.2 MB) — Xception
- `experiments/checkpoints/efficientnet/efficientnet_best.pth` (48.6 MB) — EfficientNet-B0
- `experiments/checkpoints/mobilenet/mobilenet_best.pth` (18.5 MB) — MobileNetV3-Large
- `experiments/checkpoints/vit/vit_best.pth` (260.2 MB) — Vision Transformer (ViT-B/16)
- `experiments/checkpoints/default/custom_cnn_best.pth` (1.15 MB) — Unweighted Custom CNN
- `experiments/checkpoints/class_balanced_custom_cnn/custom_cnn_best.pth` (1.15 MB) — Balanced CNN

---

## 4. Master Research Pipeline Execution

To reproduce all quantitative evaluations, master comparisons, sub-category breakdowns, statistical bootstrapping, efficiency benchmarks, image-level robustness testing, and Grad-CAM explainability, run:

```bash
python scripts/run_all_phases.py
```

### Generated Artifacts:
1. **Master Benchmark (Phase 2):**
   - `experiments/results/master_benchmark/master_comparison.csv`
   - `experiments/results/master_benchmark/master_comparison.json`
   - `experiments/results/master_benchmark/master_comparison.md`
   - `docs/images/model_comparison_bar.png`
   - `docs/images/confusion_matrix.png`
2. **Manipulation-Wise Analysis (Phase 3):**
   - `experiments/results/manipulation_analysis/manipulation_results.csv`
   - `experiments/results/manipulation_analysis/manipulation_results.json`
   - `experiments/results/manipulation_analysis/manipulation_results.md`
   - `docs/images/manipulation_breakdown.png`
3. **Quantitative Error Analysis (Phase 4):**
   - `experiments/results/resnet50_regularized/evaluation/test/errors/error_analysis_summary.json`
   - `experiments/results/resnet50_regularized/evaluation/test/errors/error_analysis_summary.md`
4. **Calibration & Reliability (Phases 5 & 6):**
   - `experiments/results/resnet50_regularized/evaluation/test/curves/calibration_summary.json`
   - `experiments/results/resnet50_regularized/evaluation/test/curves/calibration_summary.md`
5. **Bootstrap 95% Confidence Intervals (Phase 7):**
   - `experiments/results/statistical_analysis/bootstrap_results.csv`
   - `experiments/results/statistical_analysis/bootstrap_results.json`
   - `experiments/results/statistical_analysis/bootstrap_results.md`
6. **Efficiency Profiling (Phase 8):**
   - `experiments/results/efficiency/efficiency_results.csv`
   - `experiments/results/efficiency/efficiency_results.json`
   - `experiments/results/efficiency/efficiency_results.md`
   - `docs/images/efficiency_comparison.png`
7. **Robustness Evaluation (Phase 9):**
   - `experiments/results/robustness/robustness_results.csv`
   - `experiments/results/robustness/robustness_results.json`
   - `experiments/results/robustness/robustness_results.md`
   - `docs/images/robustness_curve.png`
8. **Grad-CAM Explainability (Phase 10):**
   - `experiments/results/explainability/gradcam_metadata.json`
   - `experiments/results/explainability/gradcam_summary.md`
   - `docs/images/gradcam_sample_grid.png`

---

## 5. Re-training Models (Optional)

If retraining models from scratch:
```bash
# Baseline Custom CNN (Unweighted):
python scripts/train.py --model custom_cnn --config configs/default.yaml

# Class-Balanced Custom CNN:
python scripts/train.py --model custom_cnn --pos-weight auto --output-dir experiments/results/class_balanced_custom_cnn

# ResNet50 Regularized (Strongest Model):
python scripts/train.py --model resnet50 --config configs/default.yaml --output-dir experiments/results/resnet50_regularized
```
*Note: Full model training requires a CUDA-enabled GPU (e.g., NVIDIA RTX 3080/4090 or AWS EC2 GPU instance). The evaluation and benchmarking pipeline runs deterministically on local CPU/GPU.*

---

## 6. Verification Checklist
- [x] Codebase compiles with 0 errors (`python -m compileall src scripts tests`)
- [x] 40 unit and integration tests pass (`pytest -q`)
- [x] All 8 `.pth` model checkpoints present and verified
- [x] Official test metadata verified ($N = 9,589$)
- [x] Master benchmark matrix synthesized without nested artifact errors
- [x] Manipulation analysis generated across all 7 categories
- [x] Bootstrap confidence intervals generated ($1,000$ iterations)
- [x] Robustness and Grad-CAM assets rendered in `docs/images/`
- [x] Comprehensive 29-section report in `docs/final_report.md`
