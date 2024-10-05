import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Research"))

import time
from backend.services.inference_service import predict_video

test_mp4 = Path("backend/static/forensics/fhq_7e5b3a866275.mp4")
if not test_mp4.exists():
    print("Test MP4 not found")
    sys.exit(1)

with open(test_mp4, 'rb') as f:
    video_bytes = f.read()

t0 = time.time()
res = predict_video('vit', video_bytes, '.mp4', 0.5, 24)
t1 = time.time()
print(f"Total time: {t1-t0:.2f}s")
print(f"Prediction: {res['prediction']}")
print(f"Probability: {res['probability']}")
print(f"Anomaly: {res['anomaly_type']}")
print(f"Forensic Video URL: {res['forensic_video_url']}")
print(f"Frames Analyzed: {res['framesAnalyzed']}")
