# SAM Instance Segmentation

| Item | Value |
|---|---|
| Import | `from omni_sight.instance_segmentation import SAMSegmentor` |
| Default model | `mobile_sam` (Apache-2.0) |
| Devices | `"cpu"`, `"cuda"`, `"cuda:0"` |

Model files auto-download to `~/.cache/omnisight/` on first use.

## Available model variants

### Base variants — decoder outputs 3 mask candidates

| `model_name` | Encoder | License |
|---|---|---|
| `"mobile_sam"` *(default)* | MobileSAM (ViT-Tiny) | Apache-2.0 |
| `"sam_vit_b"` | SAM ViT-Base | Apache-2.0 |
| `"sam_vit_l"` | SAM ViT-Large | Apache-2.0 |
| `"sam_vit_h"` | SAM ViT-Huge | Apache-2.0 |

### Multi-output variants — decoder returns N > 3 mask candidates

Append `-m` to any base name. The encoder is identical; only the decoder changes.

| `model_name` | Encoder |
|---|---|
| `"mobile_sam-m"` | MobileSAM (ViT-Tiny) |
| `"sam_vit_b-m"` | SAM ViT-Base |
| `"sam_vit_l-m"` | SAM ViT-Large |
| `"sam_vit_h-m"` | SAM ViT-Huge |

## Run segmentation

At least one of `point_coords` or `box` must be provided.

```python
import cv2
import numpy as np
from omni_sight.instance_segmentation import SAMSegmentor

seg = SAMSegmentor(device="cpu", model_name="mobile_sam")
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)  # RGB (H, W, 3)

# Point prompt — click foreground (label=1) or background (label=0)
masks, scores = seg.run(
    img=image,
    point_coords=np.array([[300, 200]], dtype=np.float32),  # (x, y)
    point_labels=np.array([1], dtype=np.float32),           # 1=fg, 0=bg
)

# Box prompt
masks, scores = seg.run(
    img=image,
    box=np.array([100, 80, 400, 320], dtype=np.float32),  # [x1, y1, x2, y2]
)

# Combined prompt
masks, scores = seg.run(
    img=image,
    point_coords=np.array([[300, 200]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.float32),
    box=np.array([100, 80, 400, 320], dtype=np.float32),
)
```

**Outputs**

| Variable | Shape | Content |
|---|---|---|
| `masks` | `(N, H, W)` bool | N mask candidates sorted best → worst by IoU. N=3 for base; N>3 for `-m` variants. |
| `scores` | `(N,)` float32 | IoU quality scores in `[0, 1]`; `scores[0]` is the best |

Use `masks[0]` for the best single mask.

## Use a multi-output decoder

```python
seg = SAMSegmentor(device="cpu", model_name="sam_vit_b-m")
masks, scores = seg.run(img=image, box=np.array([100, 80, 400, 320], dtype=np.float32))
# masks.shape[0] > 3
```

## Provide local ONNX files (skip auto-download)

```python
seg = SAMSegmentor(
    device="cpu",
    model_name="sam_vit_b",
    encoder_path="checkpoints/sam_vit_b_01ec64.encoder.onnx",
    decoder_path="checkpoints/sam_vit_b_01ec64.decoder.onnx",
)
```

## Visualize

```python
from omni_sight.utils.visual import draw_mask

bgr = cv2.imread("photo.jpg")
result = bgr.copy()
draw_mask(result, masks[0], color=(0, 255, 0), alpha=0.4)
cv2.imwrite("outputs/result.jpg", result)
```
