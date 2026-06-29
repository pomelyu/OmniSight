"""Command-line demo for SAM2.1 image and video segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from omni_sight.instance_segmentation import SAM2OnnxSegmentator
from omni_sight.instance_segmentation import SAM2TorchSegmentator
from omni_sight.utils.visual import draw_mask

_Segmentator = Union[SAM2OnnxSegmentator, SAM2TorchSegmentator]


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


def _open_video(path: Path):
    """Open a VideoCapture and return it with stream metadata.

    Args:
        path: Path to the video file.

    Returns:
        Tuple of ``(cap, fps, width, height, total_frames)``.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    return cap, fps, width, height, total


def _make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create an mp4v VideoWriter, making parent dirs as needed.

    Args:
        path: Output file path.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An opened :class:`cv2.VideoWriter`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def _overlay_frame(bgr: np.ndarray, mask: np.ndarray, frame_idx: int, score: Optional[float] = None) -> None:
    """Draw a green mask overlay and frame annotation on ``bgr`` in-place.

    Args:
        bgr: BGR image to annotate.
        mask: bool ``(H, W)`` mask.
        frame_idx: Frame index displayed in the annotation.
        score: Optional IoU score to append to the annotation text.
    """
    draw_mask(bgr, mask, color=(0, 255, 0), alpha=0.4)
    label = f"frame {frame_idx}" + (f"  iou={score:.3f}" if score is not None else "")
    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def _run_image(
    seg: _Segmentator,
    input_path: Path,
    output_path: Path,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
) -> None:
    """Segment a single image and save the result.

    Args:
        seg: Initialised segmentator (ONNX or PyTorch).
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


def _run_video_onnx(
    seg: SAM2OnnxSegmentator,
    input_path: Path,
    output_path: Path,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
) -> None:
    """Segment a video frame-by-frame via ONNX mask propagation.

    Args:
        seg: Initialised :class:`SAM2OnnxSegmentator`.
        input_path: Path to the input video file.
        output_path: Path to write the output ``.mp4`` video.
        point_coords: Click coordinates for the first frame or ``None``.
        point_labels: Click labels or ``None``.
        box: Bounding-box prompt for the first frame or ``None``.
    """
    cap, fps, width, height, total = _open_video(input_path)
    writer = _make_writer(output_path, fps, width, height)

    frame_idx = 0
    with tqdm(total=total, unit="frame", desc="Segmenting (ONNX)") as pbar:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if frame_idx == 0:
                masks, scores = seg.initialize(rgb, point_coords=point_coords, point_labels=point_labels, box=box)
            else:
                masks, scores = seg.propagate(rgb)
            _overlay_frame(bgr, masks[0], frame_idx, float(scores[0]))
            writer.write(bgr)
            pbar.set_postfix(iou=f"{scores[0]:.3f}")
            pbar.update(1)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames. Saved: {output_path}")


def _run_video_torch(
    seg: SAM2TorchSegmentator,
    input_path: Path,
    output_path: Path,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
) -> None:
    """Segment a video with SAM2 full temporal memory, then write annotated output.

    Loads all frames first, runs :meth:`~SAM2TorchSegmentator.process_video`
    to obtain masks for every frame, then writes the output video.

    Args:
        seg: Initialised :class:`SAM2TorchSegmentator`.
        input_path: Path to the input video file.
        output_path: Path to write the output ``.mp4`` video.
        point_coords: Click coordinates for the first frame or ``None``.
        point_labels: Click labels or ``None``.
        box: Bounding-box prompt for the first frame or ``None``.
    """
    cap, fps, width, height, total = _open_video(input_path)

    bgr_frames = []
    rgb_frames = []
    with tqdm(total=total, unit="frame", desc="Loading") as pbar:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            bgr_frames.append(bgr)
            rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pbar.update(1)
    cap.release()

    if not rgb_frames:
        raise ValueError(f"No frames read from video: {input_path}")

    print("Running SAM2 temporal propagation …")
    all_masks, _ = seg.process_video(rgb_frames, point_coords=point_coords, point_labels=point_labels, box=box)

    writer = _make_writer(output_path, fps, width, height)
    for frame_idx, (bgr, mask) in enumerate(zip(bgr_frames, all_masks)):
        _overlay_frame(bgr, mask, frame_idx)
        writer.write(bgr)
    writer.release()
    print(f"Processed {len(bgr_frames)} frames. Saved: {output_path}")


def main() -> None:
    """Run SAM2.1 segmentation from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM2.1 segmentation on an image or video.\n\n"
            "ONNX backend: fast, no PyTorch required; mask propagation only.\n"
            "PyTorch backend: full temporal memory attention; uses reference/sam2."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input image or video.")
    parser.add_argument(
        "--backend", "-b", default="onnx", choices=["onnx", "torch"],
        help="Inference backend: 'onnx' (default) or 'torch'.",
    )
    parser.add_argument("--encoder", default="", help="[ONNX] Path to the encoder ONNX file.")
    parser.add_argument("--decoder", default="", help="[ONNX] Path to the decoder ONNX file.")
    parser.add_argument("--checkpoint", default="", help="[PyTorch] Path to the .pt checkpoint file.")
    parser.add_argument(
        "--model_name", "-m", default="sam2.1_tiny",
        choices=["sam2.1_tiny", "sam2.1_small", "sam2.1_base_plus", "sam2.1_large"],
        help="Model variant (default: sam2.1_tiny).",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu).")
    parser.add_argument(
        "--points", default="",
        help='Space-separated "x,y" click coordinates for frame 0. Example: --points "320,240"',
    )
    parser.add_argument(
        "--labels", default="",
        help='Labels for --points: 1=fg, 0=bg (default: all 1). Example: --labels "1"',
    )
    parser.add_argument(
        "--box", default="",
        help='Bounding-box prompt "x1,y1,x2,y2". Example: --box "100,80,400,320"',
    )
    parser.add_argument(
        "--output", "-o", default="outputs/sam2_demo.jpg",
        help="Output path (.mp4 for video). Default: outputs/sam2_demo.jpg",
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
        point_labels = (
            np.array([float(l) for l in args.labels.split()], dtype=np.float32)
            if args.labels
            else np.ones(len(point_coords), dtype=np.float32)
        )

    box = _parse_box(args.box)

    if point_coords is None and box is None:
        raise ValueError(
            "Provide at least one prompt via --points or --box. "
            'Example: --points "320,240" --labels "1"'
        )

    seg: _Segmentator
    if args.backend == "onnx":
        seg = SAM2OnnxSegmentator(
            device=args.device,
            model_name=args.model_name,
            encoder_path=str(Path(args.encoder).resolve()) if args.encoder else None,
            decoder_path=str(Path(args.decoder).resolve()) if args.decoder else None,
        )
    else:
        if not args.checkpoint:
            raise ValueError(
                "PyTorch backend requires --checkpoint. "
                "Example: --checkpoint checkpoints/sam2.1_hiera_tiny.pt"
            )
        seg = SAM2TorchSegmentator(
            device=args.device,
            model_name=args.model_name,
            model_path=str(Path(args.checkpoint).resolve()),
        )

    is_video = input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if is_video:
        output_path = output_path.with_suffix(".mp4")
        if args.backend == "torch":
            _run_video_torch(seg, input_path, output_path, point_coords, point_labels, box)
        else:
            _run_video_onnx(seg, input_path, output_path, point_coords, point_labels, box)
    else:
        _run_image(seg, input_path, output_path, point_coords, point_labels, box)


if __name__ == "__main__":
    main()
