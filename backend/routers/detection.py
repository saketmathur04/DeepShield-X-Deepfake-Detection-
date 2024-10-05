from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from backend.services.inference_service import available_model_ids, predict_image, predict_video, predict_with_gradcam

router = APIRouter(prefix="/api", tags=["detection"])

IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/x-png", "image/pjpeg"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/avi", "video/x-matroska"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 250 * 1024 * 1024


def resolve_model(model_id: str) -> str:
    if model_id not in available_model_ids():
        raise HTTPException(status_code=422, detail="Select a trained model exposed by /api/models.")
    return model_id


async def read_upload(file: UploadFile, allowed_types: set[str], allowed_extensions: set[str], maximum_size: int) -> bytes:
    content_type = (file.content_type or "").lower()
    ext = Path(file.filename or "").suffix.lower()
    
    is_valid = (
        content_type in allowed_types
        or (content_type.startswith("image/") and ".jpg" in allowed_extensions)
        or (content_type.startswith("video/") and ".mp4" in allowed_extensions)
        or ext in allowed_extensions
    )
    if not is_valid:
        raise HTTPException(status_code=415, detail=f"Unsupported media type ({content_type or ext}). Please provide a supported image or video.")
    
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(contents) > maximum_size:
        raise HTTPException(status_code=413, detail=f"Upload exceeds the {maximum_size // 1024 // 1024} MB limit.")
    return contents

@router.post("/predict")
async def run_prediction(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(0.5)
):
    """Run real inference on an uploaded image."""
    try:
        if not 0 <= threshold <= 1:
            raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1.")
        contents = await read_upload(file, IMAGE_TYPES, IMAGE_EXTENSIONS, MAX_IMAGE_BYTES)
        result = predict_image(resolve_model(model), contents, threshold)
        return result
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/gradcam")
async def run_gradcam(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(0.5)
):
    """Run real Grad-CAM and return base64 images."""
    try:
        contents = await read_upload(file, IMAGE_TYPES, IMAGE_EXTENSIONS, MAX_IMAGE_BYTES)
        result = predict_with_gradcam(resolve_model(model), contents, threshold)
        return result
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError, ImportError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-video")
async def run_video_prediction(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float = Form(0.5),
    max_frames: int = Form(24),
):
    """Run uniform frame sampling and aggregate predictions from the selected checkpoint."""
    try:
        if not 0 <= threshold <= 1:
            raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1.")
        contents = await read_upload(file, VIDEO_TYPES, VIDEO_EXTENSIONS, MAX_VIDEO_BYTES)
        suffix = Path(file.filename or "upload.mp4").suffix
        return predict_video(resolve_model(model), contents, suffix, threshold, max_frames)
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Video inference failed. Check the server logs for details.") from exc


@router.api_route("/detection/forensic-video/{video_id}", methods=["GET", "HEAD"])
async def get_forensic_video(video_id: str):
    """Stream generated FHQ Forensic Proof Video with HTML5 range request support."""
    safe_id = Path(video_id).name
    if not safe_id.endswith(".mp4"):
        safe_id = f"{safe_id}.mp4"
    video_path = Path(__file__).resolve().parent.parent / "static" / "forensics" / safe_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Forensic proof video not found or expired.")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"DeepShield_FHQ_Proof_{safe_id}"
    )

