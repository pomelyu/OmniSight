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

| Category       | Model  | Backend | Notes                          |
|----------------|--------|---------|--------------------------------|
| Face Detection | SCRFD  | ONNX    | 3-scale and 5-scale FPN variants, optional 5-point keypoints |

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
