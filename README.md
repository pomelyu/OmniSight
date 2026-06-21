# OmniSight

A collection of portable open-source ML vision models with unified interfaces.
Focuses on image processing and face-related detection, tracking, and reconstruction.

## Installation

```bash
pip install -r requirements.txt
```

For development (includes testing, linting, and docs tools):

```bash
pip install -r requirements.dev.txt
```

## Input Convention

All OmniSight processors accept **RGB uint8** images — `np.ndarray` of shape
`(H, W, 3)`, dtype `uint8`, channel order RGB.

If loading with OpenCV (`cv2.imread`), convert before passing to any processor:

```python
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
```

## Quick Start

### Python API

```python
from omni_sight.face_detection import SCRFDFaceDetector
import cv2

detector = SCRFDFaceDetector(
    device="cpu",
    model_path="checkpoints/scrfd_10g_bnkps_shape512x512-237daff4.onnx",
)

image = cv2.imread("path/to/image.jpg")
detections, keypoints = detector.run(image)

# detections: (N, 5) array — [x1, y1, x2, y2, score]
# keypoints:  (N, 5, 2) array — 5 facial landmarks, or None
for i, det in enumerate(detections):
    x1, y1, x2, y2, score = det
    print(f"[{i}] score={score:.4f}, bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
```

### CLI Demo

```bash
# SCRFD face detection
# Download models from https://github.com/cysin/scrfd_onnx
python -m demo.demo_scrfd \
    --model scrfd_10g_bnkps \
    --image tests/resources/one_girl.jpg \
    --output outputs/result.jpg

# Options:
#   --device   cpu | cuda:0          (default: cpu)
#   --thresh   confidence threshold  (default: 0.5)
#   --nms-thresh  NMS IoU threshold  (default: 0.4)
#   --max-num  max faces to keep     (default: 0 = all)
#   --metric   center | max          (default: center)
```

## Available Models

| Category         | Model                      | Backend | Notes                                                         |
|------------------|----------------------------|---------|---------------------------------------------------------------|
| Depth Estimation | Depth Anything V2 Small    | ONNX    | Relative depth, Apache-2.0                                    |
| Depth Estimation | Depth Anything V2 Base     | ONNX    | Relative depth, CC-BY-NC-4.0                                  |
| Depth Estimation | Depth Anything V2 Large    | ONNX    | Relative depth, CC-BY-NC-4.0                                  |
| Face Detection   | SCRFD                      | ONNX    | 3-scale and 5-scale FPN variants, optional 5-point keypoints  |
| Instance Segmentation | SAM MobileSAM         | ONNX    | Prompt-guided masks (points/box), lightweight ViT-Tiny encoder, Apache-2.0 |
| Instance Segmentation | SAM ViT-B             | ONNX    | Prompt-guided masks (points/box), ViT-Base encoder, Apache-2.0 |
| Instance Segmentation | SAM ViT-L             | ONNX    | Prompt-guided masks (points/box), ViT-Large encoder, Apache-2.0 |
| Instance Segmentation | SAM ViT-H             | ONNX    | Prompt-guided masks (points/box), ViT-Huge encoder, Apache-2.0 |
| Instance Segmentation | SAM MobileSAM-multi   | ONNX    | Multi-output decoder variant of MobileSAM, returns N > 3 mask candidates |
| Instance Segmentation | SAM ViT-B-multi       | ONNX    | Multi-output decoder variant of SAM ViT-B, returns N > 3 mask candidates |
| Instance Segmentation | SAM ViT-L-multi       | ONNX    | Multi-output decoder variant of SAM ViT-L, returns N > 3 mask candidates |
| Instance Segmentation | SAM ViT-H-multi       | ONNX    | Multi-output decoder variant of SAM ViT-H, returns N > 3 mask candidates |

## Building the Documentation

```bash
cd docs
# Linux / macOS
make html

# Windows
make.bat html
```

The generated HTML is at `docs/_build/html/index.html`.

## Development

### Run Tests

```bash
pytest tests/
```
