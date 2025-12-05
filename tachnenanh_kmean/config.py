from dataclasses import dataclass

@dataclass
class SegConfig:
    # ====== CHUẨN ĐỀ BÀI (K=2, pixel-level) ======
    K: int = 2                 # Cố định 2 cụm: nền & đối tượng
    add_xy: bool = False       # CẢI TIẾN: thêm (x,y). Mặc định False để đúng baseline
    xy_weight: float = 0.40    # Trọng số (x,y) nếu bật add_xy
    blur_sigma: float = 1.0    # Gaussian σ trước K-means (khử nhiễu nhẹ)

    # K-means cấu hình
    kmeans_attempts: int = 6
    max_iter: int = 100
    eps: float = 0.5

    # Hậu xử lý
    min_hole: int = 0          # 0 = auto theo kích thước ảnh
    min_obj: int = 0           # 0 = auto theo kích thước ảnh

    # CẢI TIẾN (tùy chọn, mặc định OFF để giữ baseline thuần K-means)
    refine_gc: bool = False    # GrabCut refine

    # Chung
    max_side: int = 1600       # giới hạn chiều lớn nhất khi xử lý
    bg_mode: str = "both"      # "white" | "transparent" | "both"
