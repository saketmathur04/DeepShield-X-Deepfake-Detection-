import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Research"))

import cv2
import numpy as np
import torch
from backend.services.inference_service import get_predictor
from backend.services.face_detector import detect_and_crop_face, extract_biological_face_core

cap = cv2.VideoCapture('backend/static/forensics/fhq_4e1df32f788c.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in forensic video:", total)

frames = []
for i in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    # In forensic video, left panel is x: 25 to 625, y: 100 to 620
    # Let's extract only every 2nd or 3rd frame so we have ~30-40 frames
    if i % 2 == 0:
        left = frame[100:620, 25:625]
        frames.append((i, left))
cap.release()

print(f"Extracted {len(frames)} frames from left panel")

# Let's test with ViT and ResNet50
vit_pred = get_predictor('vit')
res_pred = get_predictor('resnet50')

vit_probs = []
res_probs = []
face_counts = 0

for idx, f in frames:
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    face_crop, bbox, has_face = detect_and_crop_face(rgb)
    target = face_crop if has_face else rgb
    if has_face:
        face_counts += 1

    with torch.no_grad():
        t_vit = vit_pred.transform(image=target)["image"].unsqueeze(0).to(vit_pred.device)
        p_vit = torch.sigmoid(vit_pred.model(t_vit).squeeze()).item()
        vit_probs.append(p_vit)

        t_res = res_pred.transform(image=target)["image"].unsqueeze(0).to(res_pred.device)
        p_res = torch.sigmoid(res_pred.model(t_res).squeeze()).item()
        res_probs.append(p_res)

print(f"Faces detected in {face_counts}/{len(frames)} frames")
print(f"ViT: min={min(vit_probs):.4f}, max={max(vit_probs):.4f}, mean={np.mean(vit_probs):.4f}, median={np.median(vit_probs):.4f}")
print(f"ResNet50: min={min(res_probs):.4f}, max={max(res_probs):.4f}, mean={np.mean(res_probs):.4f}, median={np.median(res_probs):.4f}")

# Let's also check with different face crop padding ratios (e.g. 0.25, 0.35, square)
for pad in [0.10, 0.20, 0.30, 0.40]:
    sample_crop, _, _ = detect_and_crop_face(cv2.cvtColor(frames[0][1], cv2.COLOR_BGR2RGB), pad_ratio=pad)
    with torch.no_grad():
        t_v = vit_pred.transform(image=sample_crop)["image"].unsqueeze(0).to(vit_pred.device)
        pv = torch.sigmoid(vit_pred.model(t_v).squeeze()).item()
        t_r = res_pred.transform(image=sample_crop)["image"].unsqueeze(0).to(res_pred.device)
        pr = torch.sigmoid(res_pred.model(t_r).squeeze()).item()
    print(f"Pad {pad:.2f}: ViT={pv:.4f}, ResNet={pr:.4f}")
