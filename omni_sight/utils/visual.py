from __future__ import annotations

from typing import Optional
from typing import Tuple

import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int],
    size: int,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """Draw a bounding box on an image and return the image.

    Args:
        image: BGR image array of shape ``(H, W, 3)``; modified in place.
        bbox: Bounding box in one of three formats:

            - ``[x1, y1, x2, y2]``
            - ``[x1, y1, x2, y2, score]``
            - ``[[x1, y1], [x2, y2]]``
        color: BGR colour tuple for the rectangle and score label.
        size: Thickness of the rectangle border in pixels.
        confidence: Score to display as a label. If ``None``, falls back to
            the score embedded in ``bbox`` (when present).

    Returns:
        The input ``image`` with the bounding box drawn on it.

    Raises:
        ValueError: If ``bbox`` is not one of the supported formats.
    """
    bbox_array = np.asarray(bbox)

    if bbox_array.shape == (2, 2):
        x1, y1 = bbox_array[0]
        x2, y2 = bbox_array[1]
        score = None
    elif bbox_array.ndim == 1 and bbox_array.size in (4, 5):
        x1, y1, x2, y2 = bbox_array[:4]
        score = float(bbox_array[4]) if bbox_array.size == 5 else None
    else:
        raise ValueError(
            "bbox must be [x1, y1, x2, y2], [x1, y1, x2, y2, c], "
            "or [[x1, y1], [x2, y2]]."
        )

    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        int(size),
    )

    label_score = confidence if confidence is not None else score
    if label_score is not None:
        cv2.putText(
            image,
            f"{label_score:.3f}",
            (int(x1), max(0, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return image


def draw_keypoints(
    image: np.ndarray,
    kps: np.ndarray,
    color: Tuple[int, int, int],
    size: int,
) -> np.ndarray:
    """Draw keypoints on an image and return the image.

    Args:
        image: BGR image array of shape ``(H, W, 3)``; modified in place.
        kps: Keypoint array of shape ``(N, 2)`` with ``(x, y)`` coordinates.
        color: BGR colour tuple for the keypoint circles.
        size: Radius of each keypoint circle in pixels.

    Returns:
        The input ``image`` with the keypoints drawn on it.

    Raises:
        ValueError: If ``kps`` does not have shape ``(N, 2)``.
    """
    keypoints_array = np.asarray(kps)
    if keypoints_array.ndim != 2 or keypoints_array.shape[1] != 2:
        raise ValueError("kps must have shape (N, 2).")

    for point in keypoints_array:
        px = int(point[0])
        py = int(point[1])
        cv2.circle(image, (px, py), int(size), color, -1)

    return image
