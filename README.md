# DeepShield-X

**DeepShield-X** is an end-to-end deepfake detection and facial forensics platform. It pairs a reproducible PyTorch research benchmark on **FaceForensics++ (C23)** with a full-stack web application (**FastAPI** + **React / TanStack Start**) for image and video verification, Grad-CAM visual explainability, and experiment inspection.

---

## Highlights

- **Identity-Disjoint Dataset Splitting:** Formulates FaceForensics++ (C23) partitioning as a graph connected-components problem solved with Disjoint-Set Union-Find. Across **69,273** face crops (`50,706` train / `8,978` val / `9,589` test), no actor identity or source video appears in more than one split.
- **Six-Architecture Benchmark:** Evaluates a 4-stage Custom CNN, ResNet50, Xception, EfficientNet-B0, MobileNetV3-Large, and ViT-B/16 under identical preprocessing, seed (`42`), and evaluation protocols.
- **Class Imbalance & Regularization Ablations:** Demonstrates why unweighted BCE training collapses on ~6:1 fake-to-real data (`84.40%` accuracy, `0.00%` specificity) and resolves it with inverse-frequency `pos_weight` (`0.1592`) and regularization.
- **Manipulation & Compression Audits:** Evaluates each synthesis method individually (identifying **NeuralTextures** as `55.7%` of all false negatives) and measures degradation across JPEG quality levels, noise, and blur.
- **Interactive Forensic Web App:** Upload images or videos to get frame-level/face-core predictions, Grad-CAM heatmaps, probability calibration metrics, and full experiment dashboards.

---

## System Architecture

<p align="center">
  <img src="Research/docs/images/architecture_pipeline.png" alt="DeepShield-X Pipeline" width="90%" />
</p>

---

## Benchmark Results (Held-Out Test Set, N = 9,589)

Evaluated on the frozen identity-isolated test split (`1,496` Real, `8,093` Fake) at threshold `0.50`:

| Model | Accuracy | F1 Score | Specificity | ROC-AUC | PR-AUC | MCC | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (Regularized)** | **93.89%** | **0.9633** | 87.10% | **0.9718** | **0.9938** | **0.7821** | 25.9 FPS |
| **Xception** | 93.70% | 0.9621 | **88.30%** | 0.9706 | 0.9931 | 0.7798 | 21.6 FPS |
| **EfficientNet-B0** | 92.08% | 0.9521 | 85.49% | 0.9547 | 0.9879 | 0.7290 | 35.2 FPS |
| **ViT-B/16** | 90.05% | 0.9394 | 82.75% | 0.9512 | 0.9894 | 0.6706 | 14.6 FPS |
| **MobileNetV3-Large** | 87.97% | 0.9274 | 70.99% | 0.9187 | 0.9820 | 0.5792 | **92.6 FPS** |
| **Custom CNN (Balanced)** | 43.24% | 0.5265 | 74.87% | 0.6196 | 0.8966 | 0.0987 | 122.0 FPS |
| **Custom CNN (Unweighted)** | 84.40% | 0.9154 | 0.00% | 0.5607 | 0.8799 | 0.0000 | 122.0 FPS |

<p align="center">
  <img src="Research/docs/images/gradcam_sample_grid.png" alt="Grad-CAM Saliency Heatmaps" width="85%" />
</p>

---

## Project Structure

```text
DEEPSHIELD-X/
├── backend/                # FastAPI inference & experiment API
│   ├── models/             # YuNet ONNX face detector
│   ├── routers/            # API routes (detection, models, experiments, analysis, reports)
│   ├── services/           # Inference pipeline, face-core fusion, Grad-CAM, data loader
│   ├── static/             # Generated forensic overlays (gitignored)
│   └── main.py             # FastAPI entrypoint
│
├── Frontend/               # React + TypeScript + TanStack Start UI
│   ├── public/             # Static assets and branding
│   ├── src/                # Routes, forensic visualizers, components, API client
│   └── package.json
│
└── Research/               # Training, evaluation, and academic artifacts
    ├── configs/            # YAML configs for models and experiments
    ├── data/splits/        # Identity-disjoint CSV splits
    ├── demo/               # Standalone Streamlit demo (app.py)
    ├── docs/               # Technical documentation and generated figures
    ├── experiments/        # Experiment registry, metrics JSONs, plots, checkpoint README
    ├── paper/              # LaTeX manuscript, bibliography, and project report PDF
    ├── scripts/            # Training, evaluation, splitting, and plotting scripts
    ├── src/                # Core PyTorch models, losses, Grad-CAM, and evaluation engine
    └── tests/              # Unit and integration tests
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm

### 1. Backend Setup (FastAPI)

Run the backend from the **repository root** so `backend.*` imports resolve properly:

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# Install Python dependencies
pip install -r Research/requirements.txt

# Start the FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- API Base URL: `http://127.0.0.1:8000`
- Swagger Docs: `http://127.0.0.1:8000/docs`

### 2. Frontend Setup (React + Vite)

In a second terminal:

```bash
cd Frontend
npm install
npm run dev
```

- Web App URL: `http://localhost:8080`

### 3. Model Checkpoints

Large `.pth` checkpoint files are excluded from Git tracking via `.gitignore`. See [Research/experiments/checkpoints/README.md](Research/experiments/checkpoints/README.md) for checkpoint directory layout or train models from scratch using:

```bash
cd Research
python scripts/train.py --config configs/models.yaml
```

### 4. Standalone Streamlit Demo (Optional)

```bash
cd Research
streamlit run demo/app.py
```

---

## Documentation & Paper

- **IEEE Conference Manuscript:** [Research/paper/deepshield_x_paper.tex](Research/paper/deepshield_x_paper.tex)
- **Project Report (PDF):** [Research/paper/Deepfake_Detection_Project_Report_Final.pdf](Research/paper/Deepfake_Detection_Project_Report_Final.pdf)
- **Methodology & Runbook:** [Research/docs/methodology.md](Research/docs/methodology.md) and [Research/docs/RUNBOOK.md](Research/docs/RUNBOOK.md)
- **Experiment Registry:** [Research/experiments/experiment_registry.md](Research/experiments/experiment_registry.md)

---

## License

Released under the [MIT License](Research/LICENSE).
