"""Command-line demo for Segment Anything Model (SAM) instance segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from omni_sight.instance_segmentation import SAMSegmentor
from omni_sight.utils.visual import draw_bbox
from omni_sight.utils.visual import draw_mask


def _parse_points(points_str: str) -> np.ndarray:
    """Parse a comma-separated list of ``x,y`` pairs into a ``(N, 2)`` array.

    Args:
        points_str: String like ``"100,200 300,400"`` or ``"100,200"``.

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


def main() -> None:
    """Run SAM segmentation from the command line."""
    parser = argparse.ArgumentParser(
        description="Run SAM instance segmentation on a single image."
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input image.")
    parser.add_argument(
        "--model",
        "-m",
        default="mobile_sam",
        choices=[
            "mobile_sam", "sam_vit_b", "sam_vit_l", "sam_vit_h",
            "mobile_sam-m", "sam_vit_b-m", "sam_vit_l-m", "sam_vit_h-m",
        ],
        help=(
            "Model variant (default: mobile_sam). "
            "Append -m for the multi-output decoder (more than 3 mask candidates)."
        ),
    )
    parser.add_argument(
        "--encoder",
        default="",
        help="Path to a local encoder ONNX file (bypasses auto-download).",
    )
    parser.add_argument(
        "--decoder",
        default="",
        help="Path to a local decoder ONNX file (bypasses auto-download).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, e.g. cpu or cuda:0 (default: cpu).",
    )
    parser.add_argument(
        "--points",
        default="",
        help=(
            'Space-separated "x,y" click coordinates in image pixel space. '
            'Example: --points "300,200 450,300"'
        ),
    )
    parser.add_argument(
        "--labels",
        default="",
        help=(
            "Space-separated click labels matching --points. "
            "1=foreground, 0=background. Example: --labels \"1 0\""
        ),
    )
    parser.add_argument(
        "--box",
        default="",
        help='Bounding-box prompt "x1,y1,x2,y2". Example: --box "100,80,400,320"',
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/sam_demo.jpg",
        help="Path for output image (default: outputs/sam_demo.jpg).",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
            "Example: --points \"300,200\" --labels \"1\""
        )

    seg = SAMSegmentor(
        device=args.device,
        model_name=args.model,
        encoder_path=str(Path(args.encoder).resolve()) if args.encoder else None,
        decoder_path=str(Path(args.decoder).resolve()) if args.decoder else None,
    )

    masks, iou_scores = seg.run(
        rgb,
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
    )

    print(f"Best mask IoU score: {iou_scores[0]:.4f}")
    for idx, score in enumerate(iou_scores):
        area = int(masks[idx].sum())
        print(f"[{idx}] iou={score:.4f}, area={area} px")

    result = bgr.copy()
    draw_mask(result, masks[0], color=(0, 255, 0), alpha=0.4)

    if box is not None:
        draw_bbox(result, box, color=(0, 0, 255), size=2)

    if point_coords is not None:
        for idx, (pt, lbl) in enumerate(zip(point_coords, point_labels)):
            color = (0, 255, 0) if lbl == 1.0 else (0, 0, 255)
            cv2.circle(result, (int(pt[0]), int(pt[1])), 6, color, -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved result image: {output_path}")


if __name__ == "__main__":
    main()
