"""Pixel IoU/F1 and distance-matched Pd/Fa for small-target segmentation.

Protocol and reference implementations: docs/小目标评价指标_定义与代码对照.md.
"""

import numpy as np
from scipy import ndimage


class AS1000Metrics:
    """Accumulate dataset-level metrics from probabilities and boolean GT masks.

    Predictions are thresholded at > 0.5. Components are visited in raster order;
    each GT takes the first unmatched prediction whose centroid distance is
    strictly below match_distance. Fa counts the area of unmatched prediction
    components divided by the total number of image pixels.
    """

    def __init__(self, match_distance=3, connectivity=8):
        if match_distance <= 0:
            raise ValueError('match_distance must be positive')
        if connectivity not in (4, 8):
            raise ValueError('connectivity must be 4 or 8')
        self.match_distance = match_distance
        self.connectivity = connectivity
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n_gt = 0
        self.n_detected = 0
        self.n_pred = 0
        self.false_alarm_pixels = 0
        self.total_pixels = 0
        self.n_images = 0

    def _components(self, mask):
        structure = ndimage.generate_binary_structure(2, 2 if self.connectivity == 8 else 1)
        labels, count = ndimage.label(mask, structure=structure)
        if count == 0:
            return np.empty((0, 2)), np.empty(0, dtype=np.int64)
        indices = np.arange(1, count + 1)
        centers = np.asarray(ndimage.center_of_mass(mask, labels, indices))
        areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        return centers, areas

    def update(self, prob, mask):
        """Add one native-resolution image: float probabilities and boolean GT."""
        prob, mask = np.asarray(prob), np.asarray(mask)
        if prob.ndim != 2 or prob.shape != mask.shape:
            raise ValueError('prob and mask must be 2D arrays with the same shape')
        if mask.dtype != np.bool_:
            raise ValueError('mask must be boolean; decode saved GT with gt > 128')
        if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
            raise ValueError('prob must contain finite probabilities in [0, 1]')
        binary = prob > 0.5
        self.tp += int(np.count_nonzero(binary & mask))
        self.fp += int(np.count_nonzero(binary & ~mask))
        self.fn += int(np.count_nonzero(~binary & mask))
        pred_centers, pred_areas = self._components(binary)
        gt_centers, _ = self._components(mask)
        matched = np.zeros(len(pred_centers), dtype=bool)
        for center in gt_centers:
            distances = np.linalg.norm(pred_centers - center, axis=1)
            candidates = np.flatnonzero((distances < self.match_distance) & ~matched)
            if candidates.size:
                matched[candidates[0]] = True
        self.n_gt += len(gt_centers)
        self.n_pred += len(pred_centers)
        self.n_detected += int(np.count_nonzero(matched))
        self.false_alarm_pixels += int(pred_areas[~matched].sum())
        self.total_pixels += mask.size
        self.n_images += 1

    def get(self):
        """Return raw ratios in [0, 1], with zero for zero denominators."""
        union = self.tp + self.fp + self.fn
        f1_denominator = 2 * self.tp + self.fp + self.fn
        return {
            'IoU': self.tp / union if union else 0.0,
            'F1': 2 * self.tp / f1_denominator if f1_denominator else 0.0,
            'Pd': self.n_detected / self.n_gt if self.n_gt else 0.0,
            'Fa': self.false_alarm_pixels / self.total_pixels if self.total_pixels else 0.0,
        }
