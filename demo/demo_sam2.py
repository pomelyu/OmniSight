"""Command-line demo for SAM2.1 image and video segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from omni_sight.instance_segmentation import SAM2OnnxSegmentator
from omni_sight.utils.visual import draw_mask


def _parse_points(points_str: str) -> np.ndarray:
    """Parse a space-separated list of ``x,y`` pairs into a ``(N, 2)`` array.

    Args:
        points_str: String like ``"100,200 300,400"``.

    Returns:
        float32 array of shape ``(N, 2)``.

    Raises:
        ValueError: If any token cannot be parsed as ``x,y``.
    """
    coords = []
    for token in points_str.split():
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid point '{token}'. Expected format: x,y")
        coords.append([float(parts[0]), float(parts[1])])
    return np.array(coords, dtype=np.float32)


def _parse_box(box_str: str) -> Optional[np.ndarray]:
    """Parse a ``x1,y1,x2,y2`` string into a ``(4,)`` float32 array.

    Args:
        box_str: String like ``"100,80,400,320"``.

    Returns:
        float32 array of shape ``(4,)``, or ``None`` if ``box_str`` is empty.

    Raises:
        ValueError: If the string does not contain exactly 4 values.
    """
    if not box_str:
        return None
    parts = box_str.split(",")
    if len(parts) != 4:
        raise ValueError(f"Invalid box '{box_str}'. Expected format: x1,y1,x2,y2")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _run_image(
    seg: SAM2OnnxSegmentator,
    input_path: Path,
    output_path: Path,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
) -> None:
    """Segment a single image and save the result.

    Args:
        seg: Initialised :class:`SAM2OnnxSegmentator`.
        input_path: Path to the input image.
        output_path: Path to write the output image.
        point_coords: Click coordinates or ``None``.
        point_labels: Click labels or ``None``.
        box: Bounding-box prompt or ``None``.
    """
    bgr = cv2.imread(str(input_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {input_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    masks, scores = seg.run(rgb, point_coords=point_coords, point_labels=point_labels, box=box)

    print(f"Best IoU: {scores[0]:.4f}")
    for i, score in enumerate(scores):
        print(f"  [{i}] iou={score:.4f}  area={int(masks[i].sum())} px")

    result = bgr.copy()
    draw_mask(result, masks[0], color=(0, 255, 0), alpha=0.4)
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if point_coords is not None:
        for pt, lbl in zip(point_coords, point_labels):
            color = (0, 255, 0) if lbl == 1.0 else (0, 0, 255)
            cv2.circle(result, (int(pt[0]), int(pt[1])), 6, color, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write image: {output_path}")
    print(f"Saved: {output_path}")


def _run_video(
    seg: SAM2OnnxSegmentator,
    input_path: Path,
    output_path: Path,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
) -> None:
    """Segment a video, propagating the mask across frames, and save the result.

    Args:
        seg: Initialised :class:`SAM2OnnxSegmentator`.
        input_path: Path to the input video file.
        output_path: Path to write the output ``.mp4`` video.
        point_coords: Click coordinates for the first frame or ``None``.
        point_labels: Click labels for the first frame or ``None``.
        box: Bounding-box prompt for the first frame or ``None``.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    with tqdm(total=total, unit="frame", desc="Segmenting") as pbar:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                masks, scores = seg.initialize(
                    rgb, point_coords=point_coords, point_labels=point_labels, box=box
                )
            else:
                masks, scores = seg.propagate(rgb)

            score = float(scores[0])
            draw_mask(bgr, masks[0], color=(0, 255, 0), alpha=0.4)
            cv2.putText(
                bgr,
                f"frame {frame_idx}  iou={score:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(bgr)
            pbar.set_postfix(iou=f"{score:.3f}")
            pbar.update(1)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames. Saved: {output_path}")


def main() -> None:
    """Run SAM2.1 segmentation from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM2.1 segmentation on an image or video.\n"
            "For video, the mask is propagated frame-to-frame using mask_input."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input image or video.")
    parser.add_argument(
        "--encoder",
        default="",
        help="Path to the SAM2.1 encoder ONNX file.",
    )
    parser.add_argument(
        "--decoder",
        default="",
        help="Path to the SAM2.1 decoder ONNX file.",
    )
    parser.add_argument(
        "--model_name", "-m",
        default="sam2.1_tiny",
        choices=[
            "sam2.1_tiny",
            "sam2.1_small",
            "sam2.1_base_plus",
            "sam2.1_large",
        ],
        help="Optional model variant name for validation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, e.g. cpu or cuda (default: cpu).",
    )
    parser.add_argument(
        "--points",
        default="",
        help=(
            'Space-separated "x,y" click coordinates (prompt for frame 0). '
            'Example: --points "320,240"'
        ),
    )
    parser.add_argument(
        "--labels",
        default="",
        help=(
            "Space-separated click labels matching --points. "
            "1=foreground, 0=background (default: all 1). "
            'Example: --labels "1"'
        ),
    )
    parser.add_argument(
        "--box",
        default="",
        help='Bounding-box prompt for frame 0 "x1,y1,x2,y2". Example: --box "100,80,400,320"',
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/sam2_demo.jpg",
        help="Output path. Use .mp4 extension for video output (default: outputs/sam2_demo.jpg).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    point_coords: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None
    if args.points:
        point_coords = _parse_points(args.points)
        if args.labels:
            point_labels = np.array(
                [float(l) for l in args.labels.split()], dtype=np.float32
            )
        else:
            point_labels = np.ones(len(point_coords), dtype=np.float32)

    box = _parse_box(args.box)

    if point_coords is None and box is None:
        raise ValueError(
            "Provide at least one prompt via --points or --box. "
            'Example: --points "320,240" --labels "1"'
        )

    seg = SAM2OnnxSegmentator(
        device=args.device,
        model_name=args.model_name,
        encoder_path=str(Path(args.encoder).resolve()) if args.encoder else None,
        decoder_path=str(Path(args.decoder).resolve()) if args.decoder else None,
    )

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if input_path.suffix.lower() in video_exts:
        output_path = output_path.with_suffix(".mp4")
        _run_video(seg, input_path, output_path, point_coords, point_labels, box)
    else:
        _run_image(seg, input_path, output_path, point_coords, point_labels, box)


if __name__ == "__main__":
    main()
