import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Research"))

from backend.services.inference_service import predict_video

test_mp4 = Path("backend/static/forensics/fhq_7e5b3a866275.mp4")
with open(test_mp4, 'rb') as f:
    video_bytes = f.read()

t0 = time.time()
res = predict_video('resnet50', video_bytes, '.mp4', 0.5, 24)
t1 = time.time()
print(f"Total time: {t1-t0:.2f}s")
print("prediction:", res.get("prediction"))
print("probability:", res.get("probability"))
print("confidence:", res.get("confidence"))
print("anomaly_type:", res.get("anomaly_type"))
print("forensic_video_url:", res.get("forensic_video_url"))
print("forensic_video_id:", res.get("forensic_video_id"))
if res.get("forensic_video_url"):
    print("SUCCESS! File exists:", Path("backend" + res["forensic_video_url"]).exists())
else:
    print("FAILED! forensic_video_url is None")
