import cv2
import numpy as np
from typing import Tuple, Dict, Any
from PIL import Image

from config import SegConfig
from utils import to_rgb, compose_on_white, bgr_to_rgba_pil, downscale_keep_aspect, saliency_ft, center_prior
from postprocess import postprocess
from grabcut_refine import refine_with_grabcut

# ---------- helpers ----------
def _to_lab_features(img_bgr: np.ndarray, add_xy: bool, xy_weight: float) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    feats = lab.reshape(-1, 3)
    if add_xy:
        yy, xx = np.mgrid[0:H, 0:W]
        xx = (xx.astype(np.float32) / max(W - 1, 1)).reshape(-1, 1) * xy_weight
        yy = (yy.astype(np.float32) / max(H - 1, 1)).reshape(-1, 1) * xy_weight
        feats = np.concatenate([feats, xx, yy], axis=1)
    return feats

def _kmeans_labels(img_bgr: np.ndarray, cfg: SegConfig) -> np.ndarray:
    work = img_bgr
    if cfg.blur_sigma > 0:
        work = cv2.GaussianBlur(work, (0, 0), sigmaX=cfg.blur_sigma, sigmaY=cfg.blur_sigma)
    feats = _to_lab_features(work, cfg.add_xy, cfg.xy_weight)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, cfg.max_iter, cfg.eps)
    _, labels, _ = cv2.kmeans(feats, cfg.K, None, criteria, cfg.kmeans_attempts, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(img_bgr.shape[:2])

def _edge_density(gray: np.ndarray, mask_bool: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    area = max(int(mask_bool.sum()), 1)
    return float((edges[mask_bool] > 0).sum()) / area

def _touch_border_ratio(mask_bool: np.ndarray) -> float:
    h, w = mask_bool.shape
    border = np.zeros_like(mask_bool, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    den = max(int(mask_bool.sum()), 1)
    return float((mask_bool & border).sum()) / den

def _pick_foreground_cluster(img_bgr: np.ndarray, labels: np.ndarray) -> int:
    # Chấm điểm 2 cụm theo: ít chạm viền, edge density cao, diện tích vừa phải, gần trung tâm/salient
    K = labels.max() + 1
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sal = saliency_ft(img_bgr)
    cen = center_prior(*labels.shape)
    scores = []
    for k in range(K):
        mk = (labels == k)
        area = mk.mean()
        edge = _edge_density(gray, mk)
        touch = _touch_border_ratio(mk)
        sal_mean = float(sal[mk].mean()) if mk.any() else 0.0
        cen_mean = float(cen[mk].mean()) if mk.any() else 0.0
        score = (1.6 * (1 - touch)) + (1.2 * edge) + (0.9 * sal_mean) + (0.7 * cen_mean) + (0.5 * (1 - area))
        scores.append(score)
    return int(np.argmax(scores))

def _dynamic_sizes(h: int, w: int, min_hole: int, min_obj: int):
    area = h * w
    if min_hole <= 0:
        min_hole = max(int(0.0008 * area), 32)  # 0.08% diện tích
    if min_obj <= 0:
        min_obj = max(int(0.008  * area), 128)  # 0.8%  diện tích
    return min_hole, min_obj

# ---------- baseline theo đề bài ----------
def segment_image(img_rgb: np.ndarray, cfg: SegConfig):
    if img_rgb is None:
        raise ValueError("Ảnh đầu vào trống.")
    # Giới hạn kích thước để giữ chi tiết mà không quá chậm
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr, _ = downscale_keep_aspect(img_bgr, cfg.max_side)

    # K-means với K=2, pixel-level
    labels = _kmeans_labels(img_bgr, cfg)
    k_fg = _pick_foreground_cluster(img_bgr, labels)
    raw_mask = (labels == k_fg).astype(np.uint8) * 255

    # Hậu xử lý (chuẩn hóa theo kích thước)
    h, w = raw_mask.shape
    dyn_hole, dyn_obj = _dynamic_sizes(h, w, cfg.min_hole, cfg.min_obj)
    mask = postprocess(raw_mask, dyn_hole, dyn_obj)

    # CẢI TIẾN (tuỳ chọn): GrabCut từ mask K-means
    if cfg.refine_gc:
        mask = refine_with_grabcut(img_bgr, mask, iter_count=5, min_hole=dyn_hole, min_obj=dyn_obj)

    # Xuất ảnh
    white_bgr = compose_on_white(img_bgr, mask)
    white_pil = Image.fromarray(to_rgb(white_bgr))
    transp_pil = bgr_to_rgba_pil(img_bgr, mask)
    original_pil = Image.fromarray(to_rgb(img_bgr))
    mask_pil = Image.fromarray(mask)

    # Phản hồi “đánh giá ảnh” phục vụ phần phân tích trong báo cáo
    mk = (mask > 127)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edge = _edge_density(gray, mk)
    touch = _touch_border_ratio(mk)
    # chênh màu giữa 2 tâm cụm (approx): lấy mean L*a*b của mỗi nhãn
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m0 = lab[labels == 0].mean(axis=0) if (labels == 0).any() else np.array([0,0,0])
    m1 = lab[labels == 1].mean(axis=0) if (labels == 1).any() else np.array([0,0,0])
    delta = float(np.linalg.norm(m0 - m1))
    note = (
        f"ΔLab giữa 2 cụm ≈ {delta:.1f} (càng lớn càng dễ tách) • "
        f"Edge density FG ≈ {edge:.3f} • "
        f"Tỉ lệ FG chạm viền ≈ {touch:.2f}"
    )

    meta: Dict[str, Any] = {"analysis": note}
    return original_pil, mask_pil, white_pil, transp_pil, meta
