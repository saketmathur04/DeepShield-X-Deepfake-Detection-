import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Research"))

from backend.services.inference_service import predict_video

test_mp4 = Path("backend/static/forensics/fhq_7e5b3a866275.mp4")
with open(test_mp4, 'rb') as f:
    video_bytes = f.read()

try:
    res = predict_video('resnet50', video_bytes, '.mp4', 0.5, 24)
    print("RES keys:", res.keys())
    print("forensic_video_url:", res.get("forensic_video_url"))
    print("forensic_video_id:", res.get("forensic_video_id"))
except Exception as e:
    traceback.print_exc()
