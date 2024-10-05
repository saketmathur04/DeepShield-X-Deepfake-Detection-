import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "Research"))

import cv2
import numpy as np
import torch
from backend.services.inference_service import get_predictor
from backend.services.face_detector import detect_and_crop_face, extract_biological_face_core, _get_detector

img_path = r"C:\Users\saket\Downloads\WhatsApp Image 2026-09-09 at 13.06.37.png"
img_bgr = cv2.imread(img_path)
print("Image loaded:", img_path, "Shape:", img_bgr.shape)

rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
face_crop, bbox, has_face = detect_and_crop_face(rgb)
core_crop, _, has_core = extract_biological_face_core(rgb)

cv2.imwrite("backend/scratch/ai_image_face_crop.jpg", cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
if has_core:
    cv2.imwrite("backend/scratch/ai_image_core_crop.jpg", cv2.cvtColor(core_crop, cv2.COLOR_RGB2BGR))

print(f"Face detected: {has_face}, bbox: {bbox}")
print(f"Face crop shape: {face_crop.shape if has_face else None}")
print(f"Core crop shape: {core_crop.shape if has_core else None}")

# Let's test all models across 3 targets:
# 1. Full Scene (img_array / rgb)
# 2. Face Crop
# 3. Biological Core
models = ["resnet50_regularized", "resnet50", "vit", "efficientnet", "xception", "mobilenet", "custom_cnn"]

for m in models:
    try:
        p = get_predictor(m)
        with torch.no_grad():
            # 1. Full scene
            t_full = p.transform(image=rgb)["image"].unsqueeze(0).to(p.device)
            p_full = torch.sigmoid(p.model(t_full).squeeze()).item()
            
            # 2. Face crop
            p_crop = None
            if has_face:
                t_crop = p.transform(image=face_crop)["image"].unsqueeze(0).to(p.device)
                p_crop = torch.sigmoid(p.model(t_crop).squeeze()).item()
                
            # 3. Biological core
            p_core = None
            if has_core:
                t_core = p.transform(image=core_crop)["image"].unsqueeze(0).to(p.device)
                p_core = torch.sigmoid(p.model(t_core).squeeze()).item()

        print(f"\n[{m.upper()}]")
        print(f"  Full Scene:      prob={p_full:.4f} => {'FAKE' if p_full>=0.5 else 'REAL'}")
        if p_crop is not None:
            print(f"  Face Crop:       prob={p_crop:.4f} => {'FAKE' if p_crop>=0.5 else 'REAL'}")
        if p_core is not None:
            print(f"  Biological Core: prob={p_core:.4f} => {'FAKE' if p_core>=0.5 else 'REAL'}")

    except Exception as e:
        print(f"Error testing {m}: {e}")
