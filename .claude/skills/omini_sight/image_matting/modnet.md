# MODNet Portrait Matting

| Item | Value |
|---|---|
| Import | `from omni_sight.image_matting import MODNetImageMatter` |
| Model | `modnet_photographic_portrait_matting` (~25 MB, Apache-2.0) |
| Devices | `"cpu"`, `"cuda"`, `"cuda:0"` |

Model file auto-downloads to `~/.cache/omnisight/` on first use.
Input must be **RGB uint8**, shape `(H, W, 3)`. Trained for photographic
portraits — quality degrades on non-portrait subjects.

## Run matting

```python
import cv2
from omni_sight.image_matting import MODNetImageMatter

matter = MODNetImageMatter(device="cpu")

bgr = cv2.imread("path/to/portrait.jpg")
image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB

matte = matter.run(image)
```

**Output**

| Variable | Shape | Content |
|---|---|---|
| `matte` | `(H, W)` uint8 | Alpha matte — 255 = foreground (person), 0 = background |

## Composite onto a new background

```python
import numpy as np

alpha = matte[:, :, None] / 255.0
white_bg = (bgr * alpha + 255 * (1 - alpha)).astype(np.uint8)  # BGR composite
cv2.imwrite("outputs/matted.jpg", white_bg)
```

To export a transparent PNG instead:

```python
bgra = np.dstack([bgr, matte])
cv2.imwrite("outputs/matted.png", bgra)
```
