# Model Checkpoints

Trained `.pth` checkpoint files are excluded from Git tracking in `.gitignore` because of GitHub's 100 MB file size limit (for example, `resnet50_best.pth` is 269.5 MB and `vit_best.pth` is 248.1 MB).

## Expected Directory Layout

The backend (`backend/services/inference_service.py`) and evaluation scripts look for checkpoints under `Research/experiments/checkpoints/<experiment_id>/`:

```text
Research/experiments/checkpoints/
├── resnet50_regularized/
│   └── resnet50_best.pth
├── xception/
│   └── xception_best.pth
├── efficientnet/
│   └── efficientnet_best.pth
├── vit/
│   └── vit_best.pth
├── mobilenet/
│   └── mobilenet_best.pth
├── resnet50/
│   └── resnet50_best.pth
├── class_balanced_custom_cnn/
│   └── custom_cnn_best.pth
└── default/
    └── custom_cnn_best.pth
```

## Training Checkpoints from Scratch

Once you have extracted FaceForensics++ (C23) face crops and generated the identity-disjoint splits under `Research/data/splits/`, you can train any model from `Research/`:

```bash
cd Research
python scripts/train.py --config configs/models.yaml
```

To run the full benchmark suite across all models:

```bash
cd Research
python scripts/run_experiments.py
```
