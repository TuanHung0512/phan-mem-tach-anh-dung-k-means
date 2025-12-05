import os
import gradio as gr
from typing import List

from config import SegConfig
from pipeline import segment_image
from utils import ensure_outputs_dir, timestamp_name

def build_ui():
    with gr.Blocks(title="K-means (K=2) Background Remover") as demo:

        # Thêm CSS thủ công (Gradio 6 không hỗ trợ css=)
        gr.HTML("""
        <style>
            .gr-button { 
                min-height: 44px !important; 
            }
        </style>
        """)

        gr.Markdown(
            """
            # 🖼️ K-means Background Remover — Baseline **K=2**
            Áp dụng **K-means (K=2)** để phân tách **nền** và **đối tượng** theo **pixel-level** (đúng yêu cầu đề).
            Sau khi chạy, phần *Analysis* trả về số liệu để **phân tích hiệu quả** trên ảnh nền phức tạp.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(type="numpy", label="Ảnh đầu vào", height=320)

                with gr.Accordion("Tùy chỉnh (Cải tiến — tùy chọn)", open=False):
                    add_xy = gr.Checkbox(False, label="Thêm đặc trưng (x,y) – cải tiến")
                    xy_weight = gr.Slider(0.0, 1.0, value=0.40, step=0.05, label="Trọng số (x,y)")
                    blur_sigma = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Gaussian σ")
                    min_hole = gr.Slider(0, 4000, value=0, step=20, label="Lấp lỗ nhỏ (0 = auto)")
                    min_obj  = gr.Slider(0, 20000, value=0, step=50, label="Diện tích tối thiểu (0 = auto)")
                    refine_gc = gr.Checkbox(False, label="GrabCut refine – cải tiến (KHÔNG bắt buộc)")

                    max_side = gr.Slider(640, 2400, value=1600, step=64, label="Giới hạn chiều lớn nhất")
                    bg_mode = gr.Radio(["white", "transparent", "both"], value="both", label="Xuất nền")

                start_btn = gr.Button("Start", variant="primary")
                gr.Markdown("⏳ Bấm **Start** để xử lý (sẽ hiển thị loading).")

            with gr.Column(scale=1):
                out_orig = gr.Image(type="pil", label="Original", height=240)
                out_mask = gr.Image(type="pil", label="Mask (K=2)", height=240)
                out_white = gr.Image(type="pil", label="Processed – White BG", height=240)
                out_transp = gr.Image(type="pil", label="Processed – Transparent (PNG)", height=240)
                downloads = gr.Files(label="Tải về")
                analysis = gr.Markdown(label="Analysis")

        def run_pipeline(image, add_xy_v, xy_weight_v, blur_sigma_v, min_hole_v, min_obj_v, refine_gc_v, max_side_v, bg_mode_v,
                         progress=gr.Progress(track_tqdm=False)):
            if image is None:
                raise gr.Error("Vui lòng tải ảnh lên trước.")
            progress(0.05, desc="Chuẩn bị…")

            cfg = SegConfig(
                add_xy=bool(add_xy_v),
                xy_weight=float(xy_weight_v),
                blur_sigma=float(blur_sigma_v),
                min_hole=int(min_hole_v),
                min_obj=int(min_obj_v),
                refine_gc=bool(refine_gc_v),
                max_side=int(max_side_v),
                bg_mode=str(bg_mode_v),
            )

            progress(0.6, desc="K-means (K=2) & hậu xử lý…")
            orig, mask, white, transp, meta = segment_image(image, cfg)

            progress(0.9, desc="Lưu tệp…")
            out_dir = ensure_outputs_dir("outputs")
            prefix = timestamp_name("seg_k2")
            files: List[str] = []
            if cfg.bg_mode in ("white", "both"):
                p = os.path.join(out_dir, f"{prefix}_white.png"); white.save(p); files.append(p)
            if cfg.bg_mode in ("transparent", "both"):
                p = os.path.join(out_dir, f"{prefix}_transparent.png"); transp.save(p); files.append(p)

            progress(1.0, desc="Hoàn tất")
            return orig, mask, (white if cfg.bg_mode in ("white","both") else None), \
                   (transp if cfg.bg_mode in ("transparent","both") else None), files, meta.get("analysis","")

        start_btn.click(
            fn=run_pipeline,
            inputs=[inp_img, add_xy, xy_weight, blur_sigma, min_hole, min_obj, refine_gc, max_side, bg_mode],
            outputs=[out_orig, out_mask, out_white, out_transp, downloads, analysis]
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=20).launch()
