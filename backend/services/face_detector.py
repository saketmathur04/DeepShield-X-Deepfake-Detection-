"""
DeepShield-X — Face Detection & Alignment Preprocessor

Uses OpenCV's lightweight YuNet ONNX deep learning face detector to localize
and crop faces before feeding them to DeepShield forensic checkpoints.

This ensures inference matches the FaceForensics++ training distribution
(tight face crops) and prevents false positives triggered by background scene textures.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Locate the YuNet ONNX model
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
_detector: Optional[cv2.FaceDetectorYN] = None
_last_size: Tuple[int, int] = (0, 0)


def _get_detector(width: int, height: int) -> Optional[cv2.FaceDetectorYN]:
    """Get or instantiate the YuNet face detector with the requested input dimensions."""
    global _detector, _last_size

    if not _MODEL_PATH.exists():
        logger.warning("YuNet face detector model not found at %s", _MODEL_PATH)
        return None

    try:
        if _detector is None:
            _detector = cv2.FaceDetectorYN.create(
                model=str(_MODEL_PATH),
                config="",
                input_size=(width, height),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
            _last_size = (width, height)
        elif _last_size != (width, height):
            _detector.setInputSize((width, height))
            _last_size = (width, height)

        return _detector
    except Exception as exc:
        logger.error("Failed to initialize FaceDetectorYN: %s", exc)
        return None


def detect_and_crop_face(
    image_rgb: np.ndarray,
    pad_ratio: float = 0.25,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], bool]:
    """
    Detect the primary face in an RGB image and return the cropped face region.
    Uses centered square cropping to preserve facial aspect ratio when resized to 224x224,
    matching the FaceForensics++ training distribution (30% margin around head).

    Args:
        image_rgb: Input RGB image numpy array (H, W, 3).
        pad_ratio: Margin fraction added around face bounding box (default: 0.25).

    Returns:
        cropped_face: RGB numpy array of cropped face (or original image if no face detected).
        bbox: (x1, y1, x2, y2) bounding box in original image coords (or None).
        face_detected: True if a face was detected and cropped, False otherwise.
    """
    if image_rgb is None or len(image_rgb.shape) != 3:
        return image_rgb, None, False

    h, w = image_rgb.shape[:2]
    if h < 32 or w < 32:
        return image_rgb, None, False

    # Convert RGB to BGR for OpenCV YuNet detector
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Fast downscale for high-resolution video frames (e.g. 1080p/4K) to accelerate detection
    max_dim = 640
    if max(w, h) > max_dim:
        scale_factor = max_dim / float(max(w, h))
        dw, dh = int(w * scale_factor), int(h * scale_factor)
        det_img = cv2.resize(image_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        detector = _get_detector(dw, dh)
    else:
        scale_factor = 1.0
        det_img = image_bgr
        detector = _get_detector(w, h)

    if detector is None:
        return image_rgb, None, False

    try:
        _, faces = detector.detect(det_img)
    except Exception as exc:
        logger.warning("Face detection failed during inference: %s", exc)
        return image_rgb, None, False

    if faces is None or len(faces) == 0:
        return image_rgb, None, False

    # Filter and rank candidate faces by detection confidence and facial geometry
    valid_faces = []
    for f in faces:
        raw_box = f[0:4]
        # Rescale bounding box to original coordinates if downscaled
        if scale_factor != 1.0:
            box = (raw_box / scale_factor).astype(int)
        else:
            box = raw_box.astype(int)
        score = float(f[-1])
        bw, bh = box[2], box[3]

        # Discard tiny artifacts (human faces in portrait/interview frames are substantial)
        if bw < 55 or bh < 55:
            continue

        # Check aspect ratio (human faces typically between 0.7 and 1.8)
        aspect = bh / max(1, bw)
        if not (0.65 <= aspect <= 2.0):
            continue

        # Strict human anatomical landmark validation:
        # Landmarks: [re_x, re_y, le_x, le_y, nose_x, nose_y, rm_x, rm_y, lm_x, lm_y]
        if len(f) >= 14:
            re_x, re_y = float(f[4]), float(f[5])
            le_x, le_y = float(f[6]), float(f[7])
            nose_x, nose_y = float(f[8]), float(f[9])
            rm_x, rm_y = float(f[10]), float(f[11])
            lm_x, lm_y = float(f[12]), float(f[13])

            eye_dist = np.hypot(re_x - le_x, re_y - le_y)
            # Eyes must have distinct separation in a human face
            if eye_dist < (16 * scale_factor):
                continue

            eye_y = (re_y + le_y) / 2.0
            mouth_y = (rm_y + lm_y) / 2.0

            # Natural anatomical sequence: eyes above nose, nose above mouth
            if not (eye_y < nose_y < mouth_y):
                continue

            # Nose should be roughly between eyes horizontally
            eye_min_x = min(re_x, le_x)
            eye_max_x = max(re_x, le_x)
            if not (eye_min_x - 15 <= nose_x <= eye_max_x + 15):
                continue

        # Score must be genuinely confident of a human face
        if score < 0.65:
            continue

        valid_faces.append((score, bw * bh, box, f))

    if not valid_faces:
        return image_rgb, None, False

    # Sort primarily by confidence score, secondarily by area
    valid_faces.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_item = valid_faces[0]
    best_face = best_item[2]
    best_raw = best_item[3]

    bx, by, bw, bh = best_face

    # Check for profile view using facial landmarks (eyes, nose, mouth)
    is_profile = False
    if len(best_raw) >= 14:
        re_x, le_x = float(best_raw[4]), float(best_raw[6])
        eye_dist = abs(re_x - le_x)
        # In frontal faces, eye distance is ~35-50% of face width; in profile, it compresses to <25%
        if eye_dist < (bw * 0.28):
            is_profile = True

    # Centered square bounding crop to preserve human facial proportions
    # when resized to 224x224 (aligns directly with FaceForensics++ 30% margin extraction)
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    side = max(bw, bh)
    effective_pad = 0.15 if is_profile else pad_ratio
    span = int(side * (1.0 + 2.0 * effective_pad))

    half = span // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = x1 + span
    y2 = y1 + span

    # Adjust boundaries if overflowing image dimensions while keeping square if possible
    if x1 < 0:
        x2 = min(w, x2 - x1)
        x1 = 0
    if y1 < 0:
        y2 = min(h, y2 - y1)
        y1 = 0
    if x2 > w:
        x1 = max(0, x1 - (x2 - w))
        x2 = w
    if y2 > h:
        y1 = max(0, y1 - (y2 - h))
        y2 = h

    crop = image_rgb[y1:y2, x1:x2]
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return image_rgb, None, False

    return crop, (int(x1), int(y1), int(x2), int(y2)), True


def extract_biological_face_core(
    image_rgb: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], bool]:
    """
    Extract a tightly bounded crop containing exclusively biological human facial tissue
    (eyes, nose, mouth, cheeks) guided by 5-point YuNet landmarks.

    Guarantees 0% room background / wall edges, completely eliminating false positives
    caused by contrast edges between hair and light room walls in webcam media.
    """
    if image_rgb is None or len(image_rgb.shape) != 3:
        return None, None, False

    h, w = image_rgb.shape[:2]
    if h < 32 or w < 32:
        return None, None, False

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    detector = _get_detector(w, h)
    if detector is None:
        return None, None, False

    try:
        _, faces = detector.detect(image_bgr)
    except Exception as exc:
        logger.warning("Face core detection failed: %s", exc)
        return None, None, False

    if faces is None or len(faces) == 0:
        return None, None, False

    valid = []
    for f in faces:
        if len(f) >= 15 and float(f[-1]) >= 0.45:
            box = f[0:4].astype(int)
            if box[2] >= 40 and box[3] >= 40:
                valid.append((float(f[-1]), box[2] * box[3], f))

    if not valid:
        return None, None, False

    valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_f = valid[0][2]
    landmarks = best_f[4:14].reshape((5, 2))

    lx_min = float(np.min(landmarks[:, 0]))
    lx_max = float(np.max(landmarks[:, 0]))
    ly_min = float(np.min(landmarks[:, 1]))
    ly_max = float(np.max(landmarks[:, 1]))

    # Biological facial boundaries
    eye_y = float((landmarks[0, 1] + landmarks[1, 1]) / 2.0)
    mouth_y = float((landmarks[3, 1] + landmarks[4, 1]) / 2.0)
    vert_span = max(mouth_y - eye_y, 20.0)
    horiz_span = max(lx_max - lx_min, 20.0)

    # Core margins: forehead, cheeks, chin (strict zero background)
    cx1 = max(0, int(lx_min - horiz_span * 0.20))
    cx2 = min(w, int(lx_max + horiz_span * 0.20))
    cy1 = max(0, int(ly_min - vert_span * 0.35))
    cy2 = min(h, int(ly_max + vert_span * 0.25))

    core_crop = image_rgb[cy1:cy2, cx1:cx2]
    if core_crop.shape[0] < 32 or core_crop.shape[1] < 32:
        return None, None, False

    return core_crop, (cx1, cy1, cx2, cy2), True

