"""Command-line demo for Depth Anything V2 monocular depth estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from omni_sight.depth_estimation import DepthAnythingV2Estimator
from omni_sight.utils.visual import visualize_depth

_COLORMAPS = ["inferno", "magma", "plasma", "viridis", "turbo", "gray"]


def main() -> None:
    """Run Depth Anything V2 depth estimation from the command line."""
    parser = argparse.ArgumentParser(description="Depth Anything V2 monocular depth estimation.")
    parser.add_argument("--image", "-i", required=True, help="Path to input image.")
    parser.add_argument(
        "--model", "-m",
        default="depth_anything_v2_small",
        choices=["depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large"],
        help="Model variant. Small is Apache-2.0; Base/Large are CC-BY-NC-4.0.",
    )
    parser.add_argument("--ckpt", default="", help="Path to a local ONNX model file (bypasses auto-download).")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0.")
    parser.add_argument("--output", "-o", default="outputs/depth_demo.jpg", help="Path for output depth image.")
    parser.add_argument(
        "--colormap",
        default="gray",
        choices=_COLORMAPS,
        help="Colormap applied to the depth map for visualization. Use 'gray' for grayscale.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    estimator = DepthAnythingV2Estimator(
        device=args.device,
        model_name=args.model if not args.ckpt else None,
        model_path=str(Path(args.ckpt).resolve()) if args.ckpt else None,
    )

    depth = estimator.run(image)

    print(f"Depth map shape : {depth.shape}")
    print(f"Depth range     : min={depth.min():.4f}  max={depth.max():.4f}  mean={depth.mean():.4f}")

    depth_colored = visualize_depth(depth, colormap=args.colormap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), depth_colored):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved depth image: {output_path}")


if __name__ == "__main__":
    main()
