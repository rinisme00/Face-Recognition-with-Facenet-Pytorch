# capture_faces_fullres.py
import argparse, os, json, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN

def to_pil_from_bgr(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Capture full-res frames containing faces + save raw MTCNN bboxes.")
    ap.add_argument("--camera", type=int, default=0, help="Chỉ số thiết bị camera (mặc định 0)")
    ap.add_argument("--out_dir", type=str, default="captures", help="Thư mục lưu ảnh + JSON")
    ap.add_argument("--format", type=str, default="png", choices=["png","jpg","jpeg"], help="Định dạng ảnh đầu ra")
    ap.add_argument("--jpg_quality", type=int, default=95, help="Chất lượng JPEG (nếu dùng jpg)")
    ap.add_argument("--min_conf", type=float, default=0.80, help="Ngưỡng xác suất detect tối thiểu để chấp nhận khung")
    ap.add_argument("--save_landmarks", action="store_true", help="Lưu cả landmarks của MTCNN")
    ap.add_argument("--max_frames", type=int, default=0, help="Giới hạn số frame lưu (0 = không giới hạn)")
    ap.add_argument("--no_gpu", action="store_true", help="Bắt buộc dùng CPU")
    args = ap.parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")
    print(f"[INFO] Device: {device}")

    out_dir = Path(args.out_dir)
    ensure_outdir(out_dir)
    img_dir = out_dir / "images"
    meta_dir = out_dir / "metadata"
    ensure_outdir(img_dir)
    ensure_outdir(meta_dir)

    # MTCNN: detect ở độ phân giải gốc; MTCNN tự xử lý nội bộ nhưng trả bbox theo tọa độ gốc
    mtcnn = MTCNN(keep_all=True, device=device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera.")

    # In ra độ phân giải gốc (không chỉnh sửa)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Capture @ {width}x{height} | FPS report={fps:.2f} (tuỳ driver)")

    saved = 0
    print("[INFO] Nhấn 'q' hoặc ESC để thoát.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Mất khung hình từ camera.")
            break

        pil_img = to_pil_from_bgr(frame)
        # Lấy bbox/prob (và landmarks nếu cần)
        if args.save_landmarks:
            boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
        else:
            boxes, probs = mtcnn.detect(pil_img)
            landmarks = None

        # Lưu nếu có ít nhất 1 khuôn mặt đạt min_conf
        should_save = False
        boxes_list, probs_list, lmk_list = [], [], []

        if boxes is not None and probs is not None:
            for i, (b, p) in enumerate(zip(boxes, probs)):
                if p is None:
                    continue
                if p >= args.min_conf:
                    should_save = True
                boxes_list.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                probs_list.append(float(p))
            if landmarks is not None:
                for lmk in landmarks:
                    # landmarks shape: (5,2)
                    lmk_list.append([[float(x), float(y)] for (x, y) in lmk])

        if should_save:
            ts = int(time.time() * 1000)
            img_name = f"frame_{ts}.{args.format}"
            json_name = f"frame_{ts}.json"

            img_path = str(img_dir / img_name)
            meta_path = str(meta_dir / json_name)

            if args.format.lower() in ("jpg", "jpeg"):
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
            else:
                cv2.imwrite(img_path, frame)  # PNG mặc định (lossless)

            meta = {
                "image": img_name,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "boxes": boxes_list,        # [[x1,y1,x2,y2], ...] (float) - GIỮ NGUYÊN cấu trúc MTCNN
                "probs": probs_list,        # [p1, p2, ...]
            }
            if args.save_landmarks:
                meta["landmarks"] = lmk_list  # (tuỳ chọn)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            saved += 1
            print(f"[SAVE] {img_name} (+ JSON) | faces={len(boxes_list)}")

            if args.max_frames > 0 and saved >= args.max_frames:
                print("[INFO] Đã đạt max_frames, thoát.")
                break

        # Hiển thị live (không vẽ gì để giữ nguyên khung gốc); nhấn q/ESC để thoát
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Saved {saved} frames có mặt người.")

if __name__ == "__main__":
    main()