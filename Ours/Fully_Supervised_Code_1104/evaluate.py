# 作者：恽劼恒
# 时间：2025/11/10 13:18
import os
import cv2
import argparse
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, gaussian_filter


def s_measure_folder(pred_dir, gt_dir):
    """
    Compute Structure-measure (S-measure) for all prediction/GT pairs in two folders.
    Reference: "Structure-measure: A new way to evaluate foreground maps" (ICCV 2017)

    Args:
        pred_dir (str): folder path containing predicted maps
        gt_dir (str): folder path containing ground truth maps (binary)

    Returns:
        dict: { 'mean_S': float, 'details': [(filename, score), ...] }
    """

    def _object(pred, gt):
        x = np.mean(pred[gt])
        sigma_x = np.std(pred[gt])
        score_fg = 2.0 * x / (x**2 + 1.0 + sigma_x + 1e-8)

        x = np.mean(pred[~gt])
        sigma_x = np.std(pred[~gt])
        score_bg = 2.0 * (1 - x) / ((1 - x)**2 + 1.0 + sigma_x + 1e-8)

        fg_ratio = float(np.mean(gt))
        return fg_ratio * score_fg + (1.0 - fg_ratio) * score_bg

    def _centroid(gt):
        h, w = gt.shape
        if gt.sum() == 0:
            return w // 2, h // 2
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        gt_sum = gt.sum()
        cx = np.sum(X * gt) / (gt_sum + 1e-8)
        cy = np.sum(Y * gt) / (gt_sum + 1e-8)
        return int(round(cx)), int(round(cy))

    def _divide(pred, gt, cx, cy):
        h, w = gt.shape
        X1, X2, X3, X4 = np.zeros_like(gt), np.zeros_like(gt), np.zeros_like(gt), np.zeros_like(gt)
        X1[:cy, :cx] = 1
        X2[:cy, cx:] = 1
        X3[cy:, :cx] = 1
        X4[cy:, cx:] = 1

        gt_quads = [gt * X1, gt * X2, gt * X3, gt * X4]
        pred_quads = [pred * X1, pred * X2, pred * X3, pred * X4]
        if gt.sum() == 0:
            return [pred], [gt], [1]
        w1 = gt_quads[0].sum() / gt.sum()
        w2 = gt_quads[1].sum() / gt.sum()
        w3 = gt_quads[2].sum() / gt.sum()
        w4 = gt_quads[3].sum() / gt.sum()
        return pred_quads, gt_quads, [w1, w2, w3, w4]

    def _ssim(pred, gt):
        mean_x = float(np.mean(pred))
        mean_y = float(np.mean(gt))
        std_x = float(np.std(pred))
        std_y = float(np.std(gt))
        cov = float(np.mean((pred - mean_x) * (gt - mean_y)))
        alpha = 4.0 * mean_x * mean_y * cov
        beta = (mean_x ** 2 + mean_y ** 2) * (std_x ** 2 + std_y ** 2)
        if beta == 0:
            return 1 if alpha > 0 else 0
        return alpha / (beta + 1e-8)

    def _region(pred, gt):
        cx, cy = _centroid(gt)
        pred_quads, gt_quads, weights = _divide(pred, gt, cx, cy)
        ssim_list = [_ssim(p, g) for p, g in zip(pred_quads, gt_quads)]
        return sum(w * s for w, s in zip(weights, ssim_list))

    def structure_measure(pred, gt):
        y = np.mean(gt)
        if y == 0:
            return 1 - np.mean(pred)
        elif y == 1:
            return np.mean(pred)
        else:
            alpha = 0.5
            Q = alpha * _object(pred, gt) + (1 - alpha) * _region(pred, gt)
            return max(Q, 0)

    # ---- main loop ----
    pred_files = sorted(glob(os.path.join(pred_dir, '*')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*')))

    gt_map = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in gt_files}
    scores = []

    for pf in tqdm(pred_files, desc='Computing S-measure'):
        name = os.path.splitext(os.path.basename(pf))[0].lower()
        if name not in gt_map:
            continue
        gf = gt_map[name]

        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        pred = pred.astype(np.float32) / 255.0
        gt = (gt > 128).astype(bool)

        score = structure_measure(pred, gt)
        scores.append((os.path.basename(pf), score))

    mean_score = np.mean([s for _, s in scores]) if scores else 0
    return {'mean_S': mean_score, 'details': scores}


def weighted_f_measure_folder(pred_dir, gt_dir):
    """
    Compute Weighted F-measure (WFb) for all prediction/GT pairs in two folders.
    Reference: "How to Evaluate Foreground Maps?" (Margolin et al., CVPR 2014)

    Args:
        pred_dir (str): folder path containing predicted maps
        gt_dir (str): folder path containing ground truth maps (binary)

    Returns:
        dict: { 'mean_Fw': float, 'details': [(filename, score), ...] }
    """

    def weighted_f_measure(pred, gt):
        if pred.max() > 1 or pred.min() < 0:
            raise ValueError("Prediction should be normalized to [0,1]")
        if gt.dtype != bool:
            gt = gt > 0.5

        dGT = gt.astype(np.float64)
        E = np.abs(pred - dGT)

        # Distance transform & nearest foreground indices
        Dst, IDXT = distance_transform_edt(~gt, return_indices=True)

        # Pixel dependency: propagate errors to background pixels near boundary
        Et = E.copy()
        Et[~gt] = E[tuple(IDXT[:, ~gt])]

        # Gaussian smoothing
        EA = gaussian_filter(Et, sigma=5, truncate=3.5)

        # Minimum error map
        MIN_E_EA = E.copy()
        mask = gt & (EA < E)
        MIN_E_EA[mask] = EA[mask]

        # Pixel importance
        B = np.ones_like(gt, dtype=np.float64)
        B[~gt] = 1 + (1 - np.exp(np.log(0.5) / 5 * Dst[~gt]))
        Ew = MIN_E_EA * B

        TPw = dGT.sum() - Ew[gt].sum()
        FPw = Ew[~gt].sum()

        R = 1.0 - float(np.mean(Ew[gt]))
        P = TPw / (1e-8 + TPw + FPw)

        Q = 2 * R * P / (1e-8 + R + P)
        return Q

    # ---- main loop ----
    pred_files = sorted(glob(os.path.join(pred_dir, '*')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*')))
    gt_map = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in gt_files}
    scores = []

    for pf in tqdm(pred_files, desc='Computing Weighted F-measure'):
        name = os.path.splitext(os.path.basename(pf))[0].lower()
        if name not in gt_map:
            continue
        gf = gt_map[name]

        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        pred = pred.astype(np.float32) / 255.0
        gt = (gt > 128).astype(bool)

        score = weighted_f_measure(pred, gt)
        scores.append((os.path.basename(pf), score))

    mean_score = np.mean([s for _, s in scores]) if scores else 0
    return {'mean_Fw': mean_score, 'details': scores}


def e_measure_folder(pred_dir, gt_dir):
    """
    Compute Enhanced Alignment measure (E-measure) for all prediction/GT pairs in two folders.
    Reference: "Enhanced-alignment Measure for Binary Foreground Map Evaluation"
               (Deng-Ping Fan et al., IJCAI 2018, Oral)

    Args:
        pred_dir (str): folder path containing predicted maps
        gt_dir (str): folder path containing ground truth maps (binary)

    Returns:
        dict: { 'mean_E': float, 'details': [(filename, score), ...] }
    """

    def e_measure(pred, gt):
        # Convert to double precision
        dFM = pred.astype(np.float64)
        dGT = gt.astype(np.float64)

        # Special cases
        if np.sum(dGT) == 0:  # completely black GT
            enhanced_matrix = 1.0 - dFM
        elif np.sum(~(dGT.astype(bool))) == 0:  # completely white GT
            enhanced_matrix = dFM
        else:
            # Normal case
            align_matrix = alignment_term(dFM, dGT)
            enhanced_matrix = enhanced_alignment_term(align_matrix)

        w, h = gt.shape
        score = np.sum(enhanced_matrix) / (gt.size + 1e-8)
        return score

    def alignment_term(dFM, dGT):
        mu_FM = float(np.mean(dFM))
        mu_GT = float(np.mean(dGT))
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        numerator = 2 * (align_GT * align_FM)
        denominator = align_GT ** 2 + align_FM ** 2 + 1e-8
        return numerator / denominator

    def enhanced_alignment_term(align_matrix):
        return ((align_matrix + 1) ** 2) / 4

    # ---- main loop ----
    pred_files = sorted(glob(os.path.join(pred_dir, '*')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*')))
    gt_map = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in gt_files}
    scores = []

    for pf in tqdm(pred_files, desc='Computing E-measure'):
        name = os.path.splitext(os.path.basename(pf))[0].lower()
        if name not in gt_map:
            continue
        gf = gt_map[name]

        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        # Normalize & binarize
        pred = pred.astype(np.float32) / 255.0
        gt = (gt > 128).astype(np.float32)

        score = e_measure(pred, gt)
        scores.append((os.path.basename(pf), score))

    mean_score = np.mean([s for _, s in scores]) if scores else 0
    return {'mean_E': mean_score, 'details': scores}


def mae_folder(pred_dir, gt_dir):
    """
    Compute Mean Absolute Error (MAE) for all prediction/GT pairs in two folders.
    Reference: Wangjiang Zhu et al., 2014

    Args:
        pred_dir (str): folder path containing predicted maps
        gt_dir (str): folder path containing ground truth maps (binary)

    Returns:
        dict: { 'mean_MAE': float, 'details': [(filename, score), ...] }
    """

    def calc_mae(pred, gt):
        if pred.shape != gt.shape:
            raise ValueError("Prediction and GT must have the same shape!")

        pred = pred.astype(np.float32)
        gt = gt.astype(bool)

        fg_pixels = pred[gt]
        fg_err_sum = len(fg_pixels) - np.sum(fg_pixels)
        bg_err_sum = np.sum(pred[~gt])
        mae = np.mean(np.abs(pred - gt.astype(np.float32)))
        return mae

    # ---- main loop ----
    pred_files = sorted(glob(os.path.join(pred_dir, '*')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*')))
    gt_map = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in gt_files}
    scores = []

    for pf in tqdm(pred_files, desc='Computing MAE'):
        name = os.path.splitext(os.path.basename(pf))[0].lower()
        if name not in gt_map:
            continue
        gf = gt_map[name]

        pred = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gf, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        pred = pred.astype(np.float32) / 255.0
        gt = (gt > 128).astype(bool)

        score = calc_mae(pred, gt)
        scores.append((os.path.basename(pf), score))

    mean_score = np.mean([s for _, s in scores]) if scores else 0
    return {'mean_MAE': mean_score, 'details': scores}


def evaluate_all(pred_root, gt_dir='/data/RGB-T-dataset/Seal/test/GT'):
    datasets = {
        'Seal': {
            'pred_dir': pred_root,
            'gt_dir': gt_dir
        }
    }

    results = {}
    for dataset_name, paths in datasets.items():
        pred_dir = paths['pred_dir']
        gt_dir = paths['gt_dir']

        s_res = s_measure_folder(pred_dir, gt_dir)
        f_res = weighted_f_measure_folder(pred_dir, gt_dir)
        e_res = e_measure_folder(pred_dir, gt_dir)
        mae_res = mae_folder(pred_dir, gt_dir)

        results[dataset_name] = {
            'S': float(s_res['mean_S']),
            'F': float(f_res['mean_Fw']),
            'E': float(e_res['mean_E']),
            'MAE': float(mae_res['mean_MAE']),
        }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate saliency prediction folders')
    parser.add_argument('--pred_root', type=str, default='test_maps')
    parser.add_argument('--gt_dir', type=str, default='/data/RGB-T-dataset/Seal/test/GT')
    args = parser.parse_args()

    results = evaluate_all(args.pred_root, gt_dir=args.gt_dir)
    formatted_results = {
        dataset: {metric: f"{value:.5f}" for metric, value in metrics.items()}
        for dataset, metrics in results.items()
    }
    print(json.dumps(formatted_results, ensure_ascii=False, indent=2))
