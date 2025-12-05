# K-means Background Remover (K=2)
Ứng dụng tách đối tượng khỏi phông nền bằng thuật toán K-Means với K=2.
App có giao diện Gradio: tải ảnh → bấm Start → hiển thị Original, Mask, Processed (White/Transparent), có loading và download.

Mục tiêu học thuật

Áp dụng K-Means (K=2) theo pixel để tách nền/đối tượng.

Phân tích hiệu quả trên các ảnh nền phức tạp và đưa ra gợi ý cải tiến.
## ✨ Tính năng

Baseline đúng đề: K-Means K=2, phân cụm pixel-level trên không gian Lab.

Chọn cụm foreground thông minh: phạt cụm chạm viền, ưu tiên cụm có edge density/ saliency/ center prior cao.

Hậu xử lý: morphology open/close, lấp lỗ và giữ thành phần liên thông lớn nhất (ngưỡng tỷ lệ theo kích thước ảnh).

Xuất kết quả: ảnh nền trắng, ảnh PNG trong suốt, mask nhị phân.

Phần “Analysis”: xuất các chỉ số dùng để viết báo cáo (ΔLab giữa 2 cụm, mật độ biên, tỉ lệ chạm viền).

Cải tiến (tùy chọn): thêm đặc trưng (x,y), GrabCut refine (mặc định OFF để giữ đúng baseline).
## 📦 Cấu trúc thư mục
tachnenanh_kmean/
│
├─ main.py                 # Giao diện Gradio (Start -> loading -> kết quả)
├─ pipeline.py             # Thuật toán baseline K=2 + hậu xử lý + analysis
├─ config.py               # Cấu hình tham số (K=2 cố định)
├─ utils.py                # Hàm tiện ích (IO, resize, saliency, center prior...)
├─ postprocess.py          # Morphology, fill holes, largest component
├─ grabcut_refine.py       # (Tùy chọn) refine bằng GrabCut
│
├─ outputs/                # Nơi lưu kết quả (tự tạo)
├─ samples/                # Ảnh mẫu (tự thêm 8–10 ảnh)
└─ requirements.txt

## 🛠 Cài đặt
Tạo môi trường & cài thư viện
### Khuyến nghị dùng venv
python -m venv .venv
### Windows
.venv\Scripts\activate
### macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
## ▶️ Chạy ứng dụng
    python main.py
Mở đường link Gradio hiển thị trên terminal, Upload ảnh → Start.

##🧑‍💻 Cách dùng (UI)

Upload một ảnh.

(Tuỳ chọn) mở Tùy chỉnh (Cải tiến):

Thêm đặc trưng (x,y): giúp mịn biên bằng cách thêm tọa độ pixel (mặc định OFF để đúng baseline).

GrabCut refine: tinh chỉnh biên tốt hơn (mặc định OFF).

Gaussian σ: làm mượt trước K-means (giảm nhiễu màu).

Lấp lỗ/Diện tích tối thiểu: để hậu xử lý sạch hơn (mặc định 0 = tự tính theo kích thước ảnh).

Bấm Start → đợi loading.

Xem Original, Mask (K=2), Processed – White/Transparent.

Khu vực Tải về sẽ có file PNG đã xử lý (lưu vào outputs/).


## ⚙️ Tham số quan trọng
| Tham số      | Mặc định | Ý nghĩa                                    | Gợi ý                 |
| ------------ | -------: | ------------------------------------------ | --------------------- |
| `K`          |    **2** | Số cụm K-means                             | Cố định theo đề       |
| `add_xy`     |  `False` | Thêm đặc trưng (x,y) (cải tiến)            | Bật khi biên lổn nhổn |
| `xy_weight`  |   `0.40` | Trọng số (x,y)                             | 0.3–0.6               |
| `blur_sigma` |    `1.0` | Gaussian blur trước K-means                | 0–1.5                 |
| `min_hole`   |      `0` | Lấp lỗ nhỏ (0=auto theo ảnh)               | Auto                  |
| `min_obj`    |      `0` | Diện tích tối thiểu của đối tượng (0=auto) | Auto                  |
| `refine_gc`  |  `False` | GrabCut refine (cải tiến)                  | Bật cho ảnh hơi khó   |
| `max_side`   |   `1600` | Giới hạn chiều lớn nhất khi xử lý          | 1280–2000             |
| `bg_mode`    |   `both` | Xuất nền trắng / trong suốt / cả hai       | `both` khuyến nghị    |

## 🧠 Thuật toán
Tiền xử lý: RGB → BGR, resize theo max_side, Gaussian blur nhẹ.

Đặc trưng: màu Lab (ổn định ánh sáng). (Cải tiến: thêm (x,y))

K-means (K=2): gom pixel thành nền & đối tượng.

Chọn cụm đối tượng bằng điểm tổng hợp:

phạt cụm chạm viền ảnh,

ưu tiên cụm có mật độ biên cao,

ưu tiên cụm salient và gần trung tâm,

diện tích vừa phải (không chiếm gần hết ảnh).

Hậu xử lý: morphology open/close → lấp lỗ → giữ thành phần lớn nhất (ngưỡng auto theo kích thước ảnh).

(Cải tiến – tùy chọn): GrabCut tinh chỉnh biên dựa trên mask K-means.

Kết xuất: mask, white background, transparent PNG.

