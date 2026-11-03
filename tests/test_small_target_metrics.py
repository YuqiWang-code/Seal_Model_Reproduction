import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import AS1000Metrics


class SmallTargetMetricsTests(unittest.TestCase):
    def test_perfect_and_missed_target(self):
        gt = np.zeros((12, 12), bool)
        gt[4:6, 4:6] = True
        metric = AS1000Metrics()
        metric.update(gt.astype(float), gt)
        self.assertEqual(metric.get(), dict(IoU=1.0, F1=1.0, Pd=1.0, Fa=0.0))
        metric.reset()
        metric.update(np.zeros(gt.shape), gt)
        self.assertEqual(metric.get(), dict(IoU=0.0, F1=0.0, Pd=0.0, Fa=0.0))

    def test_equal_area_false_component_is_counted(self):
        gt = np.zeros((12, 12), bool)
        gt[2:4, 2:4] = True
        pred = gt.astype(float)
        pred[8:10, 8:10] = 1
        metric = AS1000Metrics()
        metric.update(pred, gt)
        self.assertEqual(metric.get()['IoU'], 0.5)
        self.assertAlmostEqual(metric.get()['F1'], 2 / 3)
        self.assertEqual(metric.get()['Pd'], 1.0)
        self.assertEqual(metric.false_alarm_pixels, 4)
        self.assertAlmostEqual(metric.get()['Fa'], 4 / 144)

    def test_three_pixel_boundary_and_nonoverlap_match(self):
        gt = np.zeros((12, 12), bool)
        gt[5, 5] = True
        for distance, expected in [(2, 1.0), (3, 0.0)]:
            pred = np.zeros(gt.shape)
            pred[5, 5 + distance] = 1
            metric = AS1000Metrics()
            metric.update(pred, gt)
            self.assertEqual(metric.get()['Pd'], expected)
            self.assertEqual(metric.get()['IoU'], 0.0)
            self.assertEqual(metric.false_alarm_pixels, int(not expected))

    def test_one_prediction_cannot_detect_two_targets(self):
        gt = np.zeros((12, 12), bool)
        gt[5, 3] = gt[5, 7] = True
        pred = np.zeros(gt.shape)
        pred[5, 5] = 1
        metric = AS1000Metrics()
        metric.update(pred, gt)
        self.assertEqual(metric.n_gt, 2)
        self.assertEqual(metric.n_detected, 1)
        self.assertEqual(metric.get()['Pd'], 0.5)

    def test_diagonal_connectivity(self):
        gt = np.zeros((12, 12), bool)
        gt[4, 4] = gt[5, 5] = True
        eight, four = AS1000Metrics(), AS1000Metrics(connectivity=4)
        eight.update(gt.astype(float), gt)
        four.update(gt.astype(float), gt)
        self.assertEqual(eight.n_gt, 1)
        self.assertEqual(four.n_gt, 2)

    def test_global_aggregation_and_negative_image(self):
        gt = np.zeros((10, 10), bool)
        gt[4, 4] = True
        pred = gt.astype(float)
        metric = AS1000Metrics()
        metric.update(pred, gt)
        negative = np.zeros((20, 20), bool)
        false_pred = np.zeros(negative.shape)
        false_pred[2:5, 2:5] = 1
        metric.update(false_pred, negative)
        self.assertEqual(metric.get()['IoU'], 0.1)
        self.assertAlmostEqual(metric.get()['F1'], 2 / 11)
        self.assertEqual(metric.get()['Pd'], 1.0)
        self.assertEqual(metric.get()['Fa'], 9 / 500)

    def test_fixed_threshold_and_empty_dataset(self):
        gt = np.zeros((4, 4), bool)
        prob = np.zeros(gt.shape)
        prob[0, 0] = 0.5
        prob[3, 3] = 128 / 255
        metric = AS1000Metrics()
        self.assertEqual(metric.get(), dict(IoU=0.0, F1=0.0, Pd=0.0, Fa=0.0))
        metric.update(prob, gt)
        self.assertEqual(metric.fp, 1)
        self.assertEqual(metric.false_alarm_pixels, 1)


if __name__ == '__main__':
    unittest.main()
