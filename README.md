# DeepShield-X

> **Note from Developer**: This project was developed as part of my B.Tech final year project. 
> It represents months of research, experimentation, and debugging. Special thanks to my guide 
> and the open-source community for their invaluable resources.

: Enterprise & Academic DeepFake Defense System

**DeepShield-X** is an end-to-end deepfake detection and multimedia forensics platform combining academic-grade research with a production-ready web application.

---

## 🏗️ Repository Architecture

The repository is structured into three clean, decoupled subsystems:

```text
DEEPSHIELD-X/
│
├── Research/       # Academic research repository, paper, and benchmark suite
│   ├── paper/      # IEEE conference paper (LaTeX, BibTeX, Overleaf ZIP)
│   ├── src/        # Deep learning models (ResNet50, Xception, ViT, etc.)
│   ├── experiments/# Model weights (.pt checkpoints), test results, plots
│   ├── docs/       # Scientific reports, viva prep, and architecture specs
│   ├── scripts/    # Training, evaluation, and benchmark automation
│   ├── configs/    # Hyperparameter configurations (YAML)
│   ├── demo/       # Lightweight Streamlit demo (app.py)
│   └── tests/      # Automated Pytest suite
│
├── backend/        # High-performance FastAPI server
│   ├── routers/    # API endpoints (analysis, models, reports, health)
│   ├── services/   # Inference pipeline, face detection, Grad-CAM generation
│   ├── models/     # Model loader and weight management
│   ├── static/     # Static assets and forensic output storage
│   └── main.py     # FastAPI application entrypoint
│
└── Frontend/       # Modern React web interface
    ├── src/        # React components, TanStack Router, UI dashboards
    ├── public/     # Static assets, branding, and video previews
    └── package.json# Frontend dependencies and scripts
```

---

## 🚀 Quick Execution Guide

### 1. Research Subsystem
- **IEEE Paper & Overleaf:** See `Research/paper/README.md` and upload `deepshield_x_overleaf_ready.zip` to Overleaf.
- **Streamlit Demo:**
  ```bash
  cd Research
  streamlit run demo/app.py
  ```
- **Run Pytest Suite:**
  ```bash
  cd Research
  pytest -q
  ```

### 2. Full-Stack Web Application
- **Backend (FastAPI on port 8000):**
  ```bash
  cd backend
  uvicorn main:app --reload --port 8000
  ```
- **Frontend (Vite on port 8080 / 3000):**
  ```bash
  cd Frontend
  npm run dev
  ```

---

## 📄 Academic Research & Benchmark Highlights
- **Benchmark:** FaceForensics++ (FF++ C23) under a strict identity-disjoint protocol.
- **Leading Model:** ResNet50 Regularized (Accuracy: 93.89%, F1: 0.9633, ROC-AUC: 0.9718, Specificity: 87.10%).
- **Key Discovery:** Identified NeuralTextures as the primary forensic vulnerability ($55.7\%$ of false negatives).
- **Manuscript:** [Research/paper/deepshield_x_paper.tex](file:///C:/Users/saket/Desktop/DEEPSHIELD-X/DEEPSHIELD-X/Research/paper/deepshield_x_paper.tex)
