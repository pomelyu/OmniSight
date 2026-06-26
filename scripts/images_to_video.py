"""Create an MP4 video from a directory of images sorted by filename."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def images_to_video(
    image_dir: Path,
    output_path: Path,
    fps: float = 30.0,
) -> None:
    """Write all images in *image_dir* (sorted by name) to an MP4 video.

    Args:
        image_dir: Directory containing the source images.
        output_path: Destination ``.mp4`` file path.
        fps: Frames per second for the output video.

    Raises:
        FileNotFoundError: If *image_dir* does not exist.
        ValueError: If no supported images are found.
        RuntimeError: If the VideoWriter cannot be opened.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir}")

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Cannot read first image: {paths[0]}")
    h, w = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")

    for i, p in enumerate(paths):
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"  [warn] skipping unreadable frame: {p.name}")
            continue
        writer.write(frame)
        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            print(f"  {i + 1}/{len(paths)}  {p.name}")

    writer.release()
    print(f"\nSaved {len(paths)} frames → {output_path}  ({w}×{h} @ {fps} fps)")


def main() -> None:
    """Entry point for the images-to-video utility."""
    parser = argparse.ArgumentParser(
        description="Assemble images from a directory into an MP4 video."
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing input images (sorted by filename).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .mp4 path (default: <image_dir>.mp4 next to the directory).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30).",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else image_dir.parent / f"{image_dir.name}.mp4"
    )

    images_to_video(image_dir, output_path, fps=args.fps)


if __name__ == "__main__":
    main()
