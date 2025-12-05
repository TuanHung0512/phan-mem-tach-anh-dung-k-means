import cv2
import numpy as np
from postprocess import postprocess

def refine_with_grabcut(img_bgr: np.ndarray, mask_u8: np.ndarray,
                        iter_count: int = 5, min_hole: int = 64, min_obj: int = 256) -> np.ndarray:
    mask = (mask_u8 > 127).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    sure_fg = cv2.erode(mask, kernel, iterations=2)
    sure_bg = cv2.erode(1 - mask, kernel, iterations=2)

    gc_mask = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[mask == 1] = cv2.GC_PR_FGD
    gc_mask[sure_fg == 1] = cv2.GC_FGD
    gc_mask[sure_bg == 1] = cv2.GC_BGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iterCount=iter_count, mode=cv2.GC_INIT_WITH_MASK)

    refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    refined = postprocess(refined, min_hole=min_hole, min_obj=min_obj)
    return refined
