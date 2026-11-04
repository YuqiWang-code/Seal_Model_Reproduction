# 作者：恽劼恒
# 时间：2025/12/22 14:04
import os
import argparse
import cv2
import requests
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Dataset page suggests paths are within a container; this is the GCP HTTP prefix.
# If your CSV already contains full URLs, the script will use them directly.
DEFAULT_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/noaa-kotz/"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def to_url(base_url: str, path_or_url: str) -> str:
    p = str(path_or_url).strip()
    if p.lower().startswith("http://") or p.lower().startswith("https://"):
        return p
    # Ensure no leading slash duplication
    if p.startswith("/"):
        p = p[1:]
    return base_url.rstrip("/") + "/" + p


def download_file(session: requests.Session, url: str, out_path: str, timeout: int = 60) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, out_path)
        return True
    except Exception:
        return False


def read_image_cv(path: str) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        # fallback
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def sanitize_box(left, right, top, bottom) -> Optional[Tuple[float, float, float, float]]:
    """
    The dataset page notes an unusual convention: 'bottom' is the smaller y but represents logical 'top'.
    We'll just normalize by min/max so the box is well-formed in pixel coords.
    """
    try:
        l = float(left); r = float(right); t = float(top); b = float(bottom)
    except Exception:
        return None
    x1, x2 = (l, r) if l <= r else (r, l)
    y1, y2 = (t, b) if t <= b else (b, t)
    # Require positive area
    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None
    return x1, y1, x2, y2


def warp_rgb_to_ir_no_black(
    rgb_bgr: np.ndarray,
    ir_shape_hw: Tuple[int, int],
    rgb_box: Tuple[float, float, float, float],
    ir_box: Tuple[float, float, float, float],
    max_zoom: float = 8.0,
) -> Optional[np.ndarray]:
    """
    Build an affine transform: scale (sx, sy) based on box sizes, translate so box centers align.
    Then ensure no black borders in the final HxW crop by zooming in (increasing scale) if needed.
    """
    Ht, Wt = ir_shape_hw
    Hr, Wr = rgb_bgr.shape[:2]

    rx1, ry1, rx2, ry2 = rgb_box
    ix1, iy1, ix2, iy2 = ir_box

    r_w = max(1.0, rx2 - rx1)
    r_h = max(1.0, ry2 - ry1)
    i_w = max(1.0, ix2 - ix1)
    i_h = max(1.0, iy2 - iy1)

    # Non-uniform scale to match box width/height
    sx0 = i_w / r_w
    sy0 = i_h / r_h

    # Centers
    rcx = (rx1 + rx2) / 2.0
    rcy = (ry1 + ry2) / 2.0
    icx = (ix1 + ix2) / 2.0
    icy = (iy1 + iy2) / 2.0

    # After scaling: x' = sx*x + tx, y' = sy*y + ty
    # Want scaled center to land on IR center
    tx0 = icx - sx0 * rcx
    ty0 = icy - sy0 * rcy

    def corners_inside(k: float) -> bool:
        sx = sx0 * k
        sy = sy0 * k
        tx = icx - sx * rcx
        ty = icy - sy * rcy

        # Inverse-map the 4 target corners into RGB source coords
        tgt = [(0.0, 0.0), (Wt - 1.0, 0.0), (0.0, Ht - 1.0), (Wt - 1.0, Ht - 1.0)]
        for xt, yt in tgt:
            xs = (xt - tx) / sx
            ys = (yt - ty) / sy
            if xs < 0 or xs > (Wr - 1) or ys < 0 or ys > (Hr - 1):
                return False
        return True

    # Find minimal zoom k>=1 to avoid black borders (all target pixels sample inside source)
    k = 1.0
    if not corners_inside(k):
        # multiplicative search
        while k < max_zoom and not corners_inside(k):
            k *= 1.08
        if k >= max_zoom and not corners_inside(k):
            return None

    sx = sx0 * k
    sy = sy0 * k
    tx = icx - sx * rcx
    ty = icy - sy * rcy

    M = np.array([[sx, 0.0, tx],
                  [0.0, sy, ty]], dtype=np.float32)

    # warp to target size; since we ensured corners are inside, we can use any borderMode safely
    warped = cv2.warpAffine(
        rgb_bgr,
        M,
        dsize=(Wt, Ht),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # just in case of edge rounding
    )
    return warped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default='surv_test_kamera_detections_20210212_full_paths.csv', help="CSV file path (local)")
    ap.add_argument("--rgb_dir", default=r'RGB', help="Output dir for transformed RGB (.jpg)")
    ap.add_argument("--t_dir", default=r'T', help="Output dir for thermal IR (.tif)")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL,
                    help="Prefix for image paths if CSV contains relative paths")
    ap.add_argument("--start_idx", type=int, default=1, help="Start index for renaming (default 1)")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    # -----------------------
    # Logger
    # -----------------------
    log_path = os.path.join(os.getcwd(), "download.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()  # 同时打印到终端
        ]
    )

    logging.info("========== Start Processing ==========")
    logging.info(f"CSV file: {args.csv}")

    os.makedirs(args.rgb_dir, exist_ok=True)
    os.makedirs(args.t_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    logging.info(f"Total rows in CSV: {len(df)}")

    required_cols = [
        "rgb_image_path", "ir_image_path",
        "rgb_left", "rgb_right", "rgb_top", "rgb_bottom",
        "ir_left", "ir_right", "ir_top", "ir_bottom",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    sess = make_session()

    out_i = args.start_idx
    skipped = 0

    for _, row in df.iterrows():
        rgb_path = row["rgb_image_path"]
        ir_path = row["ir_image_path"]

        # Pair check
        if pd.isna(rgb_path) or pd.isna(ir_path):
            logging.warning(
                f"[SKIP][NO_PAIR] rgb={rgb_path} | ir={ir_path}"
            )
            skipped += 1
            continue

        rgb_box = sanitize_box(row["rgb_left"], row["rgb_right"], row["rgb_top"], row["rgb_bottom"])
        ir_box = sanitize_box(row["ir_left"], row["ir_right"], row["ir_top"], row["ir_bottom"])
        if rgb_box is None or ir_box is None:
            logging.warning(
                f"[SKIP][INVALID_BOX] rgb={rgb_path} | ir={ir_path}"
            )
            skipped += 1
            continue

        rgb_url = to_url(args.base_url, str(rgb_path))
        ir_url = to_url(args.base_url, str(ir_path))

        # temp download
        tmp_rgb = os.path.join(args.rgb_dir, f"__tmp_rgb_{out_i}.jpg")
        tmp_ir  = os.path.join(args.t_dir,  f"__tmp_ir_{out_i}.tif")

        ok1 = download_file(sess, rgb_url, tmp_rgb, timeout=args.timeout)
        ok2 = download_file(sess, ir_url,  tmp_ir,  timeout=args.timeout)
        if not (ok1 and ok2):
            # clean partials
            for p in [tmp_rgb, tmp_ir]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            logging.warning(
                f"[SKIP][DOWNLOAD_FAIL] rgb={rgb_url} | ir={ir_url}"
            )
            skipped += 1
            continue

        rgb = read_image_cv(tmp_rgb)
        ir  = read_image_cv(tmp_ir)
        if rgb is None or ir is None:
            for p in [tmp_rgb, tmp_ir]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            logging.warning(
                f"[SKIP][READ_FAIL] rgb={rgb_path} | ir={ir_path}"
            )
            skipped += 1
            continue

        # Ensure RGB is 3-channel BGR
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

        Ht, Wt = ir.shape[:2]
        warped_rgb = warp_rgb_to_ir_no_black(rgb, (Ht, Wt), rgb_box, ir_box)
        if warped_rgb is None:
            # can't guarantee no-black-border alignment
            for p in [tmp_rgb, tmp_ir]:
                if os.path.exists(p):
                    try: os.remove(p)
                    except: pass
            logging.warning(
                f"[SKIP][WARP_FAIL] rgb={rgb_path} | ir={ir_path}"
            )
            skipped += 1
            continue

        # Final names
        out_rgb = os.path.join(args.rgb_dir, f"{out_i}.jpg")
        out_ir  = os.path.join(args.t_dir,  f"{out_i}.tif")

        # Save: T keeps original (just rename), RGB saves warped
        # Move IR temp to final
        try:
            os.replace(tmp_ir, out_ir)
        except Exception:
            cv2.imwrite(out_ir, ir)
            try: os.remove(tmp_ir)
            except: pass

        # Write RGB warped
        cv2.imwrite(out_rgb, warped_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # remove tmp rgb
        try:
            os.remove(tmp_rgb)
        except Exception:
            pass

        if out_i % 100 == 0:
            logging.info(f"Progress: saved {out_i} pairs")

        out_i += 1

    logging.info("========== Finished ==========")
    logging.info(f"Saved pairs: {out_i - args.start_idx}")
    logging.info(f"Skipped pairs: {skipped}")


if __name__ == "__main__":
    main()