import os
import cv2
import argparse
import requests
import numpy as np
import pandas as pd
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# ============================================================
# warp_rgb_to_ir
# ============================================================
def warp_rgb_to_ir(
    rgb: np.ndarray,
    ir_shape_hw,
    rgb_box,
    ir_box,
):
    """
    Warp RGB so that:
    1) RGB box center aligns with IR box center
    2) RGB box is isotropically scaled to IR box
    3) Target is synchronously scaled
    4) Output size == IR size
    5) Black borders ARE allowed
    """
    Ht, Wt = ir_shape_hw

    rx1, ry1, rx2, ry2 = rgb_box
    tx1, ty1, tx2, ty2 = ir_box

    # --- centers ---
    rcx = (rx1 + rx2) * 0.5
    rcy = (ry1 + ry2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5

    # --- box sizes ---
    rw = max(1.0, rx2 - rx1)
    rh = max(1.0, ry2 - ry1)
    tw = max(1.0, tx2 - tx1)
    th = max(1.0, ty2 - ty1)

    # --- isotropic scale (NO extra zoom) ---
    scale = min(tw / rw, th / rh)

    # --- translation ---
    tx = tcx - scale * rcx
    ty = tcy - scale * rcy

    M = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty]
    ], dtype=np.float32)

    warped = cv2.warpAffine(
        rgb,
        M,
        dsize=(Wt, Ht),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)  # 显式黑边
    )

    return warped


# ============================================================
# utils
# ============================================================
BASE_URL = "https://storage.googleapis.com/public-datasets-lila/noaa-kotz/"


def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def to_url(path):
    if path.startswith("http"):
        return path
    return BASE_URL + path.lstrip("/")


def download_image(session, url):
    r = session.get(url, timeout=60)
    if r.status_code != 200:
        return None
    data = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def sanitize_box(l, r, t, b):
    x1, x2 = min(l, r), max(l, r)
    y1, y2 = min(t, b), max(t, b)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ============================================================
# main
# ============================================================
def main(csv_path, txt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    rgb_list = [line.strip() for line in open(txt_path) if line.strip()]

    session = make_session()
    save_idx = 1

    for rgb_rel_path in rgb_list:
        # 找到第一次出现
        rows = df[df["rgb_image_path"] == rgb_rel_path]
        if len(rows) == 0:
            continue

        row = rows.iloc[0]

        rgb_box = sanitize_box(
            row["rgb_left"], row["rgb_right"],
            row["rgb_top"], row["rgb_bottom"]
        )
        ir_box = sanitize_box(
            row["ir_left"], row["ir_right"],
            row["ir_top"], row["ir_bottom"]
        )
        if rgb_box is None or ir_box is None:
            continue

        rgb = download_image(session, to_url(row["rgb_image_path"]))
        ir = download_image(session, to_url(row["ir_image_path"]))
        if rgb is None or ir is None:
            continue

        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

        Ht, Wt = ir.shape[:2]
        warped_rgb = warp_rgb_to_ir(
            rgb, (Ht, Wt), rgb_box, ir_box
        )
        if warped_rgb is None:
            continue

        out_path = os.path.join(out_dir, f"{save_idx}.jpg")
        cv2.imwrite(out_path, warped_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[SAVE] {out_path}")

        save_idx += 1


if __name__ == "__main__":
    csv_path = 'surv_test_kamera_detections_20210212_full_paths.csv'
    txt_path = 'rgb_unique_paths.txt'
    out_dir = 'RGB'

    main(csv_path, txt_path, out_dir)
