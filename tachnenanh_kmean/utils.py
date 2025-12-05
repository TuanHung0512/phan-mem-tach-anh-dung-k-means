import os
import time
import cv2
import numpy as np
from typing import Tuple
from PIL import Image

def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def rgb_to_bgr(img_rgb):
    if img_rgb is None:
        raise ValueError("Ảnh đầu vào trống.")
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb]*3, axis=-1)
    if img_rgb.shape[2] == 4:  # RGBA -> RGB
        img_rgb = img_rgb[:, :, :3]
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def downscale_keep_aspect(img_bgr, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img_bgr, scale

def compose_on_white(img_bgr, mask_u8):
    mask = (mask_u8 > 127).astype(np.uint8)
    fg = img_bgr
    white = np.full_like(fg, 255)
    m3 = cv2.merge([mask, mask, mask])
    return np.where(m3 == 1, fg, white)

def bgr_to_rgba_pil(img_bgr, mask_u8) -> Image.Image:
    b, g, r = cv2.split(img_bgr)
    a = mask_u8
    rgba = cv2.merge([r, g, b, a])
    return Image.fromarray(rgba, mode="RGBA")

def timestamp_name(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

# ---- Saliency (Achanta FT) & center prior ----
def saliency_ft(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    blur = cv2.GaussianBlur(lab, (0, 0), 1.0)
    mean = blur.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
    s = np.linalg.norm(blur - mean, axis=2).astype(np.float32)

    # 🔧 NumPy 2.0: dùng np.ptp thay cho s.ptp()
    rng = np.ptp(s)  # = np.max(s) - np.min(s)
    if rng < 1e-6:
        return np.zeros_like(s, dtype=np.float32)
    s = (s - np.min(s)) / (rng + 1e-6)
    return s

def center_prior(h: int, w: int, sigma_ratio: float = 0.35) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    d2 = (xx - cx)**2 + (yy - cy)**2
    sigma2 = (sigma_ratio * max(h, w))**2
    prior = np.exp(-d2 / (2 * sigma2))
    return (prior - prior.min()) / (prior.max() - prior.min() + 1e-6)
