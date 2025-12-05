import cv2
import numpy as np
from skimage import morphology, measure

def morph_open_close(mask_u8: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = (mask_u8 > 127).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return (mask * 255).astype(np.uint8)

def fill_small_holes(mask_u8: np.ndarray, min_hole: int) -> np.ndarray:
    mask = (mask_u8 > 127).astype(bool)
    if min_hole > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=min_hole)
    return (mask.astype(np.uint8) * 255)

def keep_largest_component(mask_u8: np.ndarray, min_obj: int) -> np.ndarray:
    mask = (mask_u8 > 127).astype(np.uint8)
    labeled = measure.label(mask, connectivity=2)
    if labeled.max() == 0:
        return (mask * 255).astype(np.uint8)
    regions = [r for r in measure.regionprops(labeled) if r.area >= max(min_obj, 1)]
    if not regions:
        return (mask * 255).astype(np.uint8)
    best = max(regions, key=lambda r: r.area)
    keep = (labeled == best.label).astype(np.uint8)
    return (keep * 255).astype(np.uint8)

def postprocess(mask_u8: np.ndarray, min_hole: int, min_obj: int) -> np.ndarray:
    m = morph_open_close(mask_u8, 3)
    m = fill_small_holes(m, min_hole)
    m = keep_largest_component(m, min_obj)
    return m
