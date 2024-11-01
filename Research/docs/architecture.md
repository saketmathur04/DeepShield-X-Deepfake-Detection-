# DeepShield-X — System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / Config                         │
│  scripts/train.py  │  scripts/evaluate.py  │  configs/*.yaml│
└─────────┬───────────────────┬──────────────────┬────────────┘
          │                   │                  │
          ▼                   ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌────────────────────┐
│   Data Pipeline  │ │  Model Factory   │ │  Config System     │
│  ─────────────── │ │  ───────────────  │ │  ──────────────── │
│  Dataset         │ │  CustomCNN       │ │  YAML loading      │
│  Preprocessing   │ │  ResNet50        │ │  Dot-access        │
│  Face Extraction │ │  EfficientNet    │ │  CLI overrides     │
│  Splitting       │ │  MobileNet       │ │  Smoke-test mode   │
│  Validation      │ │  Xception        │ │                    │
│                  │ │  ViT             │ │                    │
└────────┬─────────┘ └────────┬─────────┘ └────────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     Training Framework                       │
│  Trainer  │  Losses  │  Optimizer  │  Scheduler  │  Checkpts │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Framework                       │
│  Metrics  │  Evaluator  │  Confusion Matrix  │  ROC/PR      │
│  Robustness  │  Error Analysis  │  Confidence Analysis       │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐
│   Explainability     │ │    Visualization     │
│  Grad-CAM            │ │  Publication plots   │
│  SHAP (optional)     │ │  Training curves     │
└──────────────────────┘ │  Comparisons         │
                         │  Robustness curves    │
                         └──────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Output Layer                          │
│  Streamlit Demo  │  Predictor (API-ready)  │  Reports        │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
src/utils/config.py     ← All modules depend on config
src/utils/seed.py       ← Training depends on seed
src/utils/device.py     ← Training/evaluation depend on device
src/utils/checkpoint.py ← Training saves, evaluation loads

src/data/*              ← Data pipeline (independent)
src/models/factory.py   ← Creates any model (depends on model modules)
src/training/trainer.py ← Uses model + data + losses + optimizer + scheduler
src/evaluation/*        ← Uses model + data + metrics
src/explainability/*    ← Uses model + pytorch-grad-cam
src/inference/predictor.py ← Uses model + preprocessing (independent of training)
```

## Data Flow

```
Raw Data → Face Extraction → Processed Faces → Metadata CSV → Source-Level Split
                                                                     │
                                            ┌────────────────────────┤
                                            ▼                        ▼
                                     Train/Val Split           Test Split
                                            │                        │
                                            ▼                        ▼
                                     Training Loop             Evaluation
                                            │                        │
                                     Checkpoints               Metrics + Plots
                                            │                        │
                                            └────────┬───────────────┘
                                                     ▼
                                              Experiment Results
```

## Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Config-driven**: No hardcoded parameters
3. **Model-agnostic**: Training pipeline works with any model from the factory
4. **Reproducible**: Seeds, environment info, and configs saved with every run
5. **Leakage-proof**: Source-level splitting with automated checks
6. **API-ready**: Predictor class is independent of UI/training code
