"""
    
DeepShield-X — Face Extraction

Face detection and cropping using OpenCV Haar cascades.
Supports both image and video inputs.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Path to OpenCV's bundled Haar cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceExtractor:
    """
    Extracts and crops faces from images or video frames.

    Uses OpenCV Haar cascades (lightweight, no extra dependencies).
    For production use, consider switching to MTCNN or dlib.
    """

    def __init__(
        self,
        output_size: int = 224,
        margin: float = 0.3,
        min_face_size: int = 30,
    ):
        """
        Args:
            output_size: Size of the cropped face image (square).
            margin: Extra margin around the detected face (fraction).
            min_face_size: Minimum face size in pixels.
        """
        self.output_size = output_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {CASCADE_PATH}")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.

        Args:
            image: BGR or RGB numpy array.

        Returns:
            List of (x, y, w, h) tuples for each detected face.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return list(faces) if len(faces) > 0 else []

    def crop_face(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
    
        Crop a face from an image with margin and resize.

        Args:
            image: BGR numpy array.
            bbox: (x, y, w, h) face bounding box.

        Returns:
            Cropped and resized face image.
        """
        x, y, w, h = bbox
        height, width = image.shape[:2]

        # Add margin
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + w + margin_x)
        y2 = min(height, y + h + margin_y)

        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (self.output_size, self.output_size))

        return face

    def extract_from_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Extract all faces from a single image.

        Args:
            image_path: Path to the image.
            output_dir: If provided, save cropped faces here.

        Returns:
            List of cropped face images.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return []

        faces_bbox = self.detect_faces(image)
        cropped_faces = []

        for i, bbox in enumerate(faces_bbox):
            face = self.crop_face(image, bbox)
            cropped_faces.append(face)

            if output_dir:
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                stem = Path(image_path).stem
                cv2.imwrite(str(out_path / f"{stem}_face_{i}.jpg"), face)

        return cropped_faces

    def extract_from_video(
        self,
        video_path: str,
        output_dir: str,
        frame_interval: int = 10,
        max_frames: int = 100,
    ) -> int:
        """
    
        Extract faces from video frames.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted face images.
            frame_interval: Extract every Nth frame.
            max_frames: Maximum number of frames to extract.

        Returns:
            Number of faces extracted.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return 0

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        face_count = 0
        video_name = Path(video_path).stem

        while cap.isOpened() and face_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                faces_bbox = self.detect_faces(frame)

                for i, bbox in enumerate(faces_bbox):
                    face = self.crop_face(frame, bbox)
                    fname = f"{video_name}_frame{frame_count:06d}_face{i}.jpg"
                    cv2.imwrite(str(out_path / fname), face)
                    face_count += 1

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {face_count} faces from {video_path} ({frame_count} frames)")
        return face_count


def batch_extract_faces(
    input_dir: str,
    output_dir: str,
    label: str = "real",
    frame_interval: int = 10,
    max_frames_per_video: int = 100,
    image_size: int = 224,
) -> int:
    """
    
    Extract faces from all videos/images in a directory.

    Args:
        input_dir: Directory containing videos or images.
        output_dir: Root output directory.
        label: Label for this batch ("real" or "fake").
        frame_interval: Extract every Nth frame from videos.
        max_frames_per_video: Max faces per video.
        image_size: Output face image size.

    Returns:
        Total number of faces extracted.
    """
    extractor = FaceExtractor(output_size=image_size)
    input_path = Path(input_dir)
    total = 0

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for item in sorted(input_path.iterdir()):
        source_name = item.stem
        source_output = Path(output_dir) / label / source_name

        if item.suffix.lower() in video_exts:
            count = extractor.extract_from_video(
                str(item), str(source_output),
                frame_interval=frame_interval,
                max_frames=max_frames_per_video,
            )
            total += count

        elif item.suffix.lower() in image_exts:
            faces = extractor.extract_from_image(str(item), str(source_output))
            total += len(faces)

        elif item.is_dir():
            # Recurse into subdirectory
            for sub_item in sorted(item.rglob("*")):
                if sub_item.suffix.lower() in video_exts:
                    sub_output = Path(output_dir) / label / sub_item.stem
                    count = extractor.extract_from_video(
                        str(sub_item), str(sub_output),
                        frame_interval=frame_interval,
                        max_frames=max_frames_per_video,
                    )
                    total += count
                elif sub_item.suffix.lower() in image_exts:
                    sub_output = Path(output_dir) / label / source_name
                    faces = extractor.extract_from_image(str(sub_item), str(sub_output))
                    total += len(faces)

    logger.info(f"Batch extraction complete: {total} faces extracted for '{label}'")
    return total
