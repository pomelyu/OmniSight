"""Command-line demo for SCRFD face detection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from omni_sight.face_detection import SCRFDFaceDetector
from omni_sight.utils.visual import draw_bbox
from omni_sight.utils.visual import draw_keypoints


def _draw_detections(
    image: np.ndarray,
    detections: np.ndarray,
    keypoints: Optional[np.ndarray],
) -> np.ndarray:
    """Draw detections by using shared visualization utilities."""
    output_image = image.copy()
    for index, det in enumerate(detections):
        draw_bbox(
            image=output_image,
            bbox=det,
            color=(0, 255, 0),
            size=5,
        )
        if keypoints is not None and index < len(keypoints):
            draw_keypoints(
                image=output_image,
                kps=keypoints[index],
                color=(255, 0, 0),
                size=5,
            )

    return output_image


def main() -> None:
    """Run SCRFD detection from command line."""
    parser = argparse.ArgumentParser(description="Run SCRFD face detection on a single image.")
    parser.add_argument("--model", "-m", required=True, help="Path to SCRFD ONNX model file.")
    parser.add_argument("--image", "-i", required=True, help="Path to input image.")
    parser.add_argument("--output", "-o", default="outputs/scrfd_demo.jpg", help="Path for output image with detections.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0.")
    parser.add_argument("--thresh", type=float, default=0.5, help="Detection confidence threshold in [0, 1].")
    parser.add_argument("--nms-thresh", type=float, default=0.4, help="NMS IoU threshold in (0, 1].")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum faces to keep. 0 means keep all.")
    parser.add_argument("--metric", choices=["center", "max"], default="center", help="Selection metric when max-num > 0.")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    detector = SCRFDFaceDetector(
        device=args.device,
        model_path=str(model_path),
    )

    detections, keypoints = detector.run(
        img=image,
        thresh=args.thresh,
        nms_thresh=args.nms_thresh,
        max_num=args.max_num,
        metric=args.metric,
    )

    print(f"Detected faces: {detections.shape[0]}")
    for index, det in enumerate(detections):
        x1, y1, x2, y2, score = det
        print(
            f"[{index}] score={score:.4f}, "
            f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_image = _draw_detections(image, detections, keypoints)
    if not cv2.imwrite(str(output_path), rendered_image):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved result image: {output_path}")


if __name__ == "__main__":
    main()
