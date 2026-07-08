from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from omni_sight.image_matting import MODNetImageMatter


def main() -> None:
    """Run MODNet portrait matting from the command line."""
    parser = argparse.ArgumentParser(description="Run MODNet image matting on a single image.")
    parser.add_argument("--image", "-i", required=True, help="Path to input image.")
    parser.add_argument("--ckpt", default="", help="Path to MODNet ONNX model file.")
    parser.add_argument(
        "--output", "-o", default="outputs/modnet_demo.jpg",
        help="Path for output image composited on a white background.",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0.")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    processor = MODNetImageMatter(
        device=args.device,
        model_path=str(Path(args.ckpt).resolve()) if args.ckpt else None,
    )

    matte = processor.run(image)

    alpha = matte[:, :, None] / 255.0
    result = (bgr * alpha + 255 * (1 - alpha)).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved result image: {output_path}")


if __name__ == "__main__":
    main()
