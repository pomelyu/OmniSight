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
### CLI Demo

```bash
# Face Detection
## SCRFD face detection (scrfd_2.5g_bnkps, scrfd_10g_bnkps)
python -m demo.demo_scrfd -i tests/resources/one_girl.jpg -m scrfd_10g_bnkps

# Depth Estimation
## depth anything v2 (depth_anything_v2_small, depth_anything_v2_base, depth_anything_v2_large)
python -m demo.demo_depth_anything_v2 -i tests/resources/one_girl.jpg -m depth_anything_v2_small

# Instance Segmentation
## SAM (mobile_sam, sam_vit_b, sam_vit_l, sam_vit_h) - return only highest confidence mask
##     (mobile_sam-m, sam_vit_b-m, sam_vit_l-m, sam_vit_h-m) - return subpart, part, whole mask
python -m demo.demo_sam -i tests/resources/pokemons.jpg --points "100,460" --labels "1" -m mobile_sam
python -m demo.demo_sam -i tests/resources/pokemons.jpg --points "100,460" --labels "1" -m mobile_sam-m

## SAM2 (sam2.1_tiny, sam2.1_small, sam2.1_base_plus, sam2.1_large) - return only highest confidence mask
python -m demo.demo_sam2 -i tests/resources/pokemons.jpg --points "100,460" --labels "1"
# video segmentation via feeding the previous frame prediction(different from original paper)
python -m demo.demo_sam2 -i tests/resources/pikabear.mp4 --points "580,140" --labels "1"

```

## Available Models

| Category         | Model                      | Backend | Notes                                                         |
|------------------|----------------------------|---------|---------------------------------------------------------------|
| Depth Estimation | Depth Anything V2 Small    | ONNX    | Relative depth, Apache-2.0                                    |
| Depth Estimation | Depth Anything V2 Base     | ONNX    | Relative depth, CC-BY-NC-4.0                                  |
| Depth Estimation | Depth Anything V2 Large    | ONNX    | Relative depth, CC-BY-NC-4.0                                  |
| Face Detection   | SCRFD                      | ONNX    | 3-scale and 5-scale FPN variants, optional 5-point keypoints  |
| Instance Segmentation | SAM2.1 Hiera Tiny     | ONNX    | Image/video segmentation via mask propagation, ViT-Tiny backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Small    | ONNX    | Image/video segmentation via mask propagation, ViT-Small backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Base+    | ONNX    | Image/video segmentation via mask propagation, ViT-Base+ backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Large    | ONNX    | Image/video segmentation via mask propagation, ViT-Large backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Tiny     | PyTorch | Image/video segmentation with full temporal memory attention, ViT-Tiny backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Small    | PyTorch | Image/video segmentation with full temporal memory attention, ViT-Small backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Base+    | PyTorch | Image/video segmentation with full temporal memory attention, ViT-Base+ backbone, Apache-2.0 |
| Instance Segmentation | SAM2.1 Hiera Large    | PyTorch | Image/video segmentation with full temporal memory attention, ViT-Large backbone, Apache-2.0 |
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
