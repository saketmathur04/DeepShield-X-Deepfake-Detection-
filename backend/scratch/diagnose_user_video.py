import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Research"))

import cv2
import numpy as np
import torch
from backend.services.inference_service import get_predictor
from backend.services.face_detector import detect_and_crop_face, extract_biological_face_core, _get_detector

# Read frame from latest forensic video
cap = cv2.VideoCapture('backend/static/forensics/fhq_4e1df32f788c.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read forensic video")
    sys.exit(1)

# Left panel is roughly x: 25 to 625, y: 100 to 620
# Let's crop the left panel out:
left_panel = frame[100:620, 25:625]
cv2.imwrite('backend/scratch/left_panel.jpg', left_panel)
print("Saved left_panel.jpg, shape:", left_panel.shape)

# Detect face in left panel
rgb = cv2.cvtColor(left_panel, cv2.COLOR_BGR2RGB)
detector = _get_detector(rgb.shape[1], rgb.shape[0])
_, faces = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("Detected faces:", len(faces) if faces is not None else 0)

if faces is not None and len(faces) > 0:
    box = faces[0][:4].astype(int)
    print("Box:", box)
    bx, by, bw, bh = box
    
    # 1. Tight crop (pad=0.10)
    p_tight = 0.10
    c_tight = rgb[max(0, by-int(bh*p_tight)):min(rgb.shape[0], by+bh+int(bh*p_tight)),
                  max(0, bx-int(bw*p_tight)):min(rgb.shape[1], bx+bw+int(bw*p_tight))]
    cv2.imwrite('backend/scratch/crop_tight.jpg', cv2.cvtColor(c_tight, cv2.COLOR_RGB2BGR))

    # 2. Training distribution crop (pad=0.30)
    p_30 = 0.30
    c_30 = rgb[max(0, by-int(bh*p_30)):min(rgb.shape[0], by+bh+int(bh*p_30)),
               max(0, bx-int(bw*p_30)):min(rgb.shape[1], bx+bw+int(bw*p_30))]
    cv2.imwrite('backend/scratch/crop_30.jpg', cv2.cvtColor(c_30, cv2.COLOR_RGB2BGR))

    # 3. Square centered crop with 0.35 margin (captures full head like FaceForensics++)
    side = max(bw, bh)
    cx, cy = bx + bw//2, by + bh//2
    half = int(side * 0.75)  # 1.5x side total
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(rgb.shape[1], cx + half), min(rgb.shape[0], cy + half)
    c_sq = rgb[y1:y2, x1:x2]
    cv2.imwrite('backend/scratch/crop_square.jpg', cv2.cvtColor(c_sq, cv2.COLOR_RGB2BGR))

    # 4. Biological core
    core, _, _ = extract_biological_face_core(rgb)
    if core is not None:
        cv2.imwrite('backend/scratch/crop_core.jpg', cv2.cvtColor(core, cv2.COLOR_RGB2BGR))

    # Now let's test predictors across models!
    models = ['vit', 'resnet50', 'efficientnet', 'xception']
    for m in models:
        try:
            pred = get_predictor(m)
            print(f"\n--- MODEL: {m} ---")
            
            # Predict with tight crop
            r_tight = pred.predict(c_tight)
            print(f"Tight crop (pad 0.10): prob={r_tight['probability']} label={r_tight['label']}")
            
            # Predict with 0.30 crop
            r_30 = pred.predict(c_30)
            print(f"Margin 0.30 crop:      prob={r_30['probability']} label={r_30['label']}")

            # Predict with square crop
            r_sq = pred.predict(c_sq)
            print(f"Square 1.5x crop:      prob={r_sq['probability']} label={r_sq['label']}")

            # Predict with core
            if core is not None:
                r_core = pred.predict(core)
                print(f"Biological core:       prob={r_core['probability']} label={r_core['label']}")

            # What if image is passed as BGR vs RGB?
            r_bgr = pred.predict(cv2.cvtColor(c_30, cv2.COLOR_RGB2BGR))
            print(f"0.30 crop passed BGR:  prob={r_bgr['probability']} label={r_bgr['label']}")

        except Exception as e:
            print(f"Error testing {m}: {e}")
