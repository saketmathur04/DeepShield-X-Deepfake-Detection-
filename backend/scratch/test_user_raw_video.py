import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "Research"))

import cv2
import numpy as np
import torch
from backend.services.inference_service import get_predictor, available_model_ids
from backend.services.face_detector import detect_and_crop_face, extract_biological_face_core, _get_detector

raw_video_path = r"C:\Users\saket\AppData\Local\Temp\tmp8sft6l53.mp4"
print("Testing raw user video:", raw_video_path, "Size:", os.path.getsize(raw_video_path))

cap = cv2.VideoCapture(raw_video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video props: {w}x{h}, {frame_count} frames, {fps:.1f} fps")

# Sample 24 frames uniformly
target_indices = sorted({round(i * (frame_count - 1) / 23) for i in range(24)})
frames = []
for idx in target_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames.append((idx, frame))
cap.release()

print(f"Sampled {len(frames)} frames")

# Check face detector on first frame
sample_bgr = frames[0][1]
sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
detector = _get_detector(w, h)
_, raw_faces = detector.detect(sample_bgr)
print(f"Detector found {len(raw_faces) if raw_faces is not None else 0} raw faces on first frame")
if raw_faces is not None and len(raw_faces) > 0:
    print("Primary face box:", raw_faces[0][:4].astype(int), "score:", float(raw_faces[0][-1]))

# Test each model on the user's video frames!
models_to_test = ['resnet50', 'efficientnet', 'mobilenet', 'custom_cnn', 'xception', 'vit']

for mid in models_to_test:
    try:
        pred = get_predictor(mid)
        probs_tight = []
        probs_30 = []
        probs_core = []
        
        for idx, f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            
            # 1. Tight crop (pad 0.10)
            c_tight, _, has_face = detect_and_crop_face(rgb, pad_ratio=0.10)
            t_tight = pred.transform(image=c_tight if has_face else rgb)["image"].unsqueeze(0).to(pred.device)
            p_tight = torch.sigmoid(pred.model(t_tight).squeeze()).item()
            probs_tight.append(p_tight)

            # 2. 0.30 margin crop
            c_30, _, _ = detect_and_crop_face(rgb, pad_ratio=0.30)
            t_30 = pred.transform(image=c_30 if has_face else rgb)["image"].unsqueeze(0).to(pred.device)
            p_30 = torch.sigmoid(pred.model(t_30).squeeze()).item()
            probs_30.append(p_30)

            # 3. Biological face core
            core, _, has_core = extract_biological_face_core(rgb)
            if has_core and core is not None:
                t_core = pred.transform(image=core)["image"].unsqueeze(0).to(pred.device)
                p_core = torch.sigmoid(pred.model(t_core).squeeze()).item()
                probs_core.append(p_core)

        print(f"\n================ MODEL: {mid.upper()} ================")
        print(f"  Tight (0.10): mean={np.mean(probs_tight):.4f}, median={np.median(probs_tight):.4f}, min={np.min(probs_tight):.4f}, max={np.max(probs_tight):.4f}")
        print(f"  Pad 0.30:     mean={np.mean(probs_30):.4f}, median={np.median(probs_30):.4f}, min={np.min(probs_30):.4f}, max={np.max(probs_30):.4f}")
        if probs_core:
            print(f"  Bio Core:     mean={np.mean(probs_core):.4f}, median={np.median(probs_core):.4f}, min={np.min(probs_core):.4f}, max={np.max(probs_core):.4f}")

    except Exception as exc:
        print(f"Error testing {mid}: {exc}")
