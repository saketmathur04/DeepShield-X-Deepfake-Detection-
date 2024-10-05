import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "Research"))

from backend.services.inference_service import predict_video, predict_image
import cv2

print("=== 1. VERIFYING USER AUTHENTIC FRAME (predict_image) ===")
with open("backend/scratch/left_panel.jpg", "rb") as f:
    img_bytes = f.read()

for m in ["resnet50_regularized", "resnet50", "xception", "vit"]:
    res = predict_image(m, img_bytes, threshold=0.5)
    print(f"[{m:20s}] Verdict: {res['prediction']:4s} | Prob: {res['probability']:.4f} | Confidence: {res['confidence']*100:.1f}% | Face Detected: {res['face_detected']}")

print("\n=== 2. VERIFYING KNOWN DEEPFAKE TEST IMAGE (predict_image) ===")
fake_path = "Research/data/faces/test/fake/Deepfakes/040_997/frame_0000.jpg"
if os.path.exists(fake_path):
    with open(fake_path, "rb") as f:
        fake_bytes = f.read()
    for m in ["resnet50_regularized", "resnet50", "xception", "vit"]:
        res_fake = predict_image(m, fake_bytes, threshold=0.5)
        print(f"[{m:20s}] Verdict: {res_fake['prediction']:4s} | Prob: {res_fake['probability']:.4f} | Confidence: {res_fake['confidence']*100:.1f}%")
else:
    print("Fake sample path not found on disk")

print("\n=== 3. VERIFYING VIDEO PREDICTION (predict_video) ===")
# Test with forensic sample video
vid_path = "backend/static/forensics/fhq_7e5b3a866275.mp4"
if os.path.exists(vid_path):
    with open(vid_path, "rb") as f:
        vb = f.read()
    for m in ["resnet50_regularized", "resnet50"]:
        v_res = predict_video(m, vb, ".mp4", threshold=0.5, max_frames=24)
        print(f"[{m:20s}] Video Verdict: {v_res['prediction']:4s} | Prob: {v_res['probability']:.4f} | Confidence: {v_res['confidence']*100:.1f}% | Video Generated: {bool(v_res['forensic_video_url'])}")
