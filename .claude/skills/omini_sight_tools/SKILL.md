---
name: omini_sight_tools
description: >
  OmniSight utility helpers for image processing, visualization, file downloading,
  and hashing. Use this skill whenever the user asks to draw bounding boxes or
  keypoints, download a model file, verify a file's hash, or use any omni_sight.utils
  helper — even if they don't mention OmniSight by name.
---

# OmniSight Utility Helpers

## Visualization (`omni_sight.utils.visual`)

```python
from omni_sight.utils.visual import draw_bbox, draw_keypoints
import numpy as np

image = np.zeros((512, 512, 3), dtype=np.uint8)

# bbox accepts [x1, y1, x2, y2], [x1, y1, x2, y2, score], or [[x1,y1],[x2,y2]]
draw_bbox(image=image, bbox=np.array([10, 10, 100, 100]), color=(0, 255, 0), size=2)

# kps shape: (K, 2)
draw_keypoints(image=image, kps=np.random.rand(5, 2) * 512, color=(255, 0, 0), size=3)
```

## File downloading (`omni_sight.utils.file_loader`)

```python
from omni_sight.utils.file_loader import download_url_to_file
from pathlib import Path

download_url_to_file(
    url="https://example.com/model.onnx",
    destination=Path("checkpoints/model.onnx"),
    hash_prefix="bfd8deac",   # optional SHA-256 prefix for integrity check
)
```

## Hashing (`omni_sight.utils.hash`)

```python
from omni_sight.utils.hash import get_sha256_hash, parse_file_hash

digest = get_sha256_hash("checkpoints/model.onnx")   # full 64-char hex string
prefix = parse_file_hash("scrfd_10g_bnkps-731cbbfd.onnx")  # → "731cbbfd"
```
