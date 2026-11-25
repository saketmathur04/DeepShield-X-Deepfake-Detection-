import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup plotting style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style="whitegrid", font_scale=1.2)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
CHECKPOINTS_DIR = ROOT / "experiments" / "checkpoints"
IMAGES_DIR = ROOT / "docs" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def plot_model_comparison():
    master_json = RESULTS_DIR / "master_benchmark" / "master_comparison.json"
    if not master_json.exists():
        print("Master JSON not found.")
        return
    with open(master_json) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        primary = data.get("primary_comparison", {})
    elif isinstance(data, list):
        primary = {item.get("display_name", item.get("model_id")): item for item in data}
    else:
        return
    
    if not primary:
        return
    
    models = []
    f1_scores = []
    roc_aucs = []
    
    # Sort models by F1
    sorted_models = sorted(primary.items(), key=lambda x: x[1].get('f1', 0.0) or 0.0, reverse=False)
    for model_name, metrics in sorted_models:
        name_clean = model_name.replace("_", " ").title()
        models.append(name_clean)
        f1_scores.append((metrics.get('f1', 0.0) or 0.0) * 100)
        roc_aucs.append((metrics.get('roc_auc', 0.0) or 0.0) * 100)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = np.arange(len(models))
    height = 0.35
    
    rects1 = ax.barh(y - height/2, f1_scores, height, label='F1 Score (%)', color='#2b8cbe')
    rects2 = ax.barh(y + height/2, roc_aucs, height, label='ROC-AUC (%)', color='#f03b20')
    
    ax.set_xlabel('Score (%)', fontweight='bold')
    ax.set_title('Primary Architecture Comparison on FaceForensics++ (C23)', fontweight='bold', pad=20)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(50, 105)
    
    # Add labels with explicit percentages
    for rects in [rects1, rects2]:
        for rect in rects:
            width = rect.get_width()
            ax.annotate(f'{width:.2f}%',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(4, 0),  # 4 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "model_comparison_bar.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print("Saved model_comparison_bar.png")

def plot_training_history():
    base_path = CHECKPOINTS_DIR / "resnet50" / "resnet50_history.json"
    reg_path = CHECKPOINTS_DIR / "resnet50_regularized" / "resnet50_history.json"
    
    if not base_path.exists() or not reg_path.exists():
        print("History files not found.")
        return
        
    with open(base_path) as f:
        h_base = json.load(f)
    with open(reg_path) as f:
        h_reg = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Dynamics: Baseline vs. Regularized (ResNet50)', fontweight='bold', fontsize=14)
    
    epochs = list(range(1, len(h_base['train_loss']) + 1))
    epochs_reg = list(range(1, len(h_reg['train_loss']) + 1))
    
    # ------------------------------------------------------------------
    # 1. Loss Over Epochs
    # ------------------------------------------------------------------
    ax1.plot(epochs, h_base['train_loss'], color='#b0c4de', linestyle='-', label='Baseline Train')
    ax1.plot(epochs, h_base['val_loss'], color='#4c72b0', linestyle='--', marker='p', label='Baseline Val')
    
    ax1.plot(epochs_reg, h_reg['train_loss'], color='#a1d99b', linestyle='-', label='Regularized Train')
    ax1.plot(epochs_reg, h_reg['val_loss'], color='#41ab5d', linestyle='--', marker='s', label='Regularized Val')
    
    ax1.set_title('Loss Over Epochs', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # ------------------------------------------------------------------
    # 2. Accuracy Over Epochs
    # ------------------------------------------------------------------
    ax2.plot(epochs, h_base['train_acc'], color='#b0c4de', linestyle='-', label='Baseline Train Acc')
    ax2.plot(epochs, h_base['val_acc'], color='#4c72b0', linestyle='--', marker='p', label='Baseline Val Acc')
    
    ax2.plot(epochs_reg, h_reg['train_acc'], color='#a1d99b', linestyle='-', label='Regularized Train Acc')
    ax2.plot(epochs_reg, h_reg['val_acc'], color='#41ab5d', linestyle='--', marker='s', label='Regularized Val Acc')
    
    ax2.set_title('Accuracy Over Epochs', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "training_dynamics.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print("Saved training_dynamics.png")

def plot_confusion_matrix():
    actual_real = 1496
    actual_fake = 8093
    
    recall = 0.9514
    specificity = 0.8710
    
    tp = int(actual_fake * recall)
    fn = actual_fake - tp
    tn = int(actual_real * specificity)
    fp = actual_real - tn
    
    cm = np.array([[tn, fp], [fn, tp]])
    annot_cm = np.array([
        [f"{tn:,}\n({tn/actual_real*100:.1f}% Spec)", f"{fp:,}\n({fp/actual_real*100:.1f}% FPR)"],
        [f"{fn:,}\n({fn/actual_fake*100:.1f}% FNR)", f"{tp:,}\n({tp/actual_fake*100:.1f}% Rec)"]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot_cm, fmt='', cmap='Blues', 
                xticklabels=['REAL (0)', 'FAKE (1)'],
                yticklabels=['REAL (0)', 'FAKE (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title('Confusion Matrix: ResNet50 (Regularized)\n[Counts & Percentage Rates]', fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print("Saved confusion_matrix.png")
    plt.close()
    print("Saved confusion_matrix.png")

if __name__ == "__main__":
    plot_model_comparison()
    plot_training_history()
    plot_confusion_matrix()
