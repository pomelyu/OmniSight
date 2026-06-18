# Depth Anything V2

| Item | Value |
|---|---|
| Import | `from omni_sight.depth_estimation import DepthAnythingV2Estimator` |
| Default model | `depth_anything_v2_small` (~99 MB, Apache-2.0) |
| Other models | `depth_anything_v2_base` (389 MB), `depth_anything_v2_large` (~1.3 GB) — CC-BY-NC-4.0 |
| Devices | `"cpu"`, `"cuda"`, `"cuda:0"` |

Model files auto-download to `~/.cache/omnisight/` on first use.
Input must be **RGB uint8**, shape `(H, W, 3)`.

## Run estimation

```python
import cv2
from omni_sight.depth_estimation import DepthAnythingV2Estimator

estimator = DepthAnythingV2Estimator(device="cpu", model_name="depth_anything_v2_small")

bgr = cv2.imread("path/to/image.jpg")
image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB

depth = estimator.run(image)
```

**Output**

| Variable | Shape | Content |
|---|---|---|
| `depth` | `(H, W)` float32 | Relative depth — larger values = farther from camera |

## Visualize

```python
from omni_sight.utils.visual import visualize_depth
import cv2

vis = visualize_depth(depth)                        # grayscale (default)
vis = visualize_depth(depth, colormap="inferno")    # false-color

cv2.imwrite("outputs/depth.jpg", vis)               # vis is BGR uint8
```

Supported colormaps: `"gray"` (default), `"inferno"`, `"magma"`, `"plasma"`, `"viridis"`, `"turbo"`.
