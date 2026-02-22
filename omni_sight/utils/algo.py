from typing import List

import numpy as np


def nms(dets: np.ndarray, thresh: float) -> List[int]:
    """Apply greedy NMS and return kept indices.

    This function is kept reusable at module scope.
    """
    if dets.size == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)

        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        inter = width * height
        overlap = inter / (areas[idx] + areas[order[1:]] - inter)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    return keep
