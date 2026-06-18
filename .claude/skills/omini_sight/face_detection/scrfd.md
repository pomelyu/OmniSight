# SCRFD Face Detection

| Item | Value |
|---|---|
| Import | `from omni_sight.face_detection import SCRFDFaceDetector` |
| Default model | `scrfd_10g_bnkps` (~100 MB, higher accuracy) |
| Light model | `scrfd_2.5g_bnkps` (~20 MB) |
| Devices | `"cpu"`, `"cuda"`, `"cuda:0"` |

Model files auto-download to `checkpoints/` on first use.

## Run detection

```python
import cv2
from omni_sight.face_detection import SCRFDFaceDetector

detector = SCRFDFaceDetector(device="cpu", model_name="scrfd_10g_bnkps")
image = cv2.imread("path/to/image.jpg")  # BGR (H, W, 3)

detections, keypoints = detector.run(
    img=image,
    thresh=0.5,      # lower → more recall; raise → fewer false positives
    nms_thresh=0.4,
    max_num=0,       # 0 = return all faces
    metric="center", # "center" | "max" — used only when max_num > 0
)
```

**Outputs**

| Variable | Shape | Content |
|---|---|---|
| `detections` | `(K, 5)` float32 | `[x1, y1, x2, y2, score]` per face |
| `keypoints` | `(K, 5, 2)` float32 or `None` | left eye, right eye, nose, left mouth, right mouth |

`detections` is `(0, 5)` when no faces pass the threshold.

## Visualize

```python
from omni_sight.utils.visual import draw_bbox, draw_keypoints

out = image.copy()
for i, det in enumerate(detections):
    draw_bbox(image=out, bbox=det, color=(0, 255, 0), size=2)
    if keypoints is not None:
        draw_keypoints(image=out, kps=keypoints[i], color=(255, 0, 0), size=3)
cv2.imwrite("outputs/result.jpg", out)
```
