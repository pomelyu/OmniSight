---
name: omini_sight
description: >
  OmniSight models for image and face processing. Use this skill whenever
  the user asks to detect faces, estimate depth, segment objects, run any
  vision model, or process images through OmniSight. For utility helpers
  (visualization, file loading, hashing), use the omini_sight_tools skill.
---

# OmniSight Model Index

Read the relevant sub-skill file for full API details.
All processors accept **RGB uint8** images — shape `(H, W, 3)`, dtype `uint8`.

## Instance Segmentation

| Model | Notes | File |
|---|---|---|
| SAM (MobileSAM / ViT-B) | Prompt-guided binary masks from points or bounding box | `instance_segmentation/sam.md` |

## Depth Estimation

| Model | Notes | File |
|---|---|---|
| Depth Anything V2 | Monocular relative depth, Small/Base/Large variants | `depth_estimation/depth_anything_v2.md` |

## Image Matting

| Model | Notes | File |
|---|---|---|
| MODNet | Portrait alpha matting, returns uint8 alpha map | `image_matting/modnet.md` |

## Face Detection

| Model | Notes | File |
|---|---|---|
| SCRFD | Fast ONNX detector, returns bounding boxes + 5-pt landmarks | `face_detection/scrfd.md` |
