import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from pathlib import Path

# Setup
plt.rcParams['figure.dpi'] = 300
IMAGES_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def draw_box(ax, x, y, width, height, text, bg_color='#a6cee3', text_color='black', fontsize=12):
    # Draw rounded rectangle
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.1,rounding_size=0.1",
                                 ec="black", fc=bg_color, lw=1.5)
    ax.add_patch(box)
    
    # Add text
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', color=text_color, fontsize=fontsize,
            fontweight='bold', fontfamily='sans-serif')

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=15))

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(8.5, 5.5, 'DeepShield-X Detection Pipeline', ha='center', va='center', 
            fontsize=20, fontweight='bold')
    
    # Pipeline stages
    y_center = 3
    box_w = 1.9
    box_h = 1.5
    gap = 0.7
    
    stages = [
        ("Raw Video\n(C23)", '#fdbf6f'),
        ("Union-Find\nIdentity Split", '#b2df8a'),
        ("Face Cropping\n(224x224)", '#1f78b4'),
        ("Transfer\nLearning\n(ResNet50)", '#fb9a99'),
        ("Grad-CAM\nExplainability", '#cab2d6'),
        ("Binary\nClassification\n(Real/Fake)", '#e31a1c')
    ]
    
    x_curr = 0.6
    centers = []
    
    for i, (text, color) in enumerate(stages):
        # The main text color is white for darker backgrounds, black for lighter
        text_color = 'white' if color in ['#1f78b4', '#e31a1c'] else 'black'
        draw_box(ax, x_curr, y_center - box_h/2, box_w, box_h, text, bg_color=color, text_color=text_color)
        centers.append((x_curr + box_w, y_center))
        x_curr += box_w + gap
        
    # Draw arrows
    for i in range(len(centers) - 1):
        start_x = centers[i][0]
        end_x = start_x + gap
        draw_arrow(ax, start_x, y_center, end_x, y_center)
        
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "architecture_pipeline.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print("Saved architecture_pipeline.png")

if __name__ == "__main__":
    generate_architecture_diagram()

