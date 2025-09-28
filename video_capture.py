# apps/crop_mtcnn.py
import argparse, os
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def list_images(p: Path, recursive: bool):
    if p.is_file(): return [p]
    it = p.rglob("*") if recursive else p.glob("*")
    return [q for q in it if q.suffix.lower() in IMG_EXTS]

def tensor_to_bgr_u8(face_tensor):
    """
    Nhận Tensor [3,H,W] (RGB), có thể nằm ở [-1,1] hoặc [0,1] hoặc [0,255].
    Trả về ảnh BGR uint8 (0..255) để cv2.imwrite.
    """
    x = face_tensor.detach().cpu()
    mn, mx = float(x.min()), float(x.max())
    if mn >= -1.1 and mx <= 1.1:                # chuẩn hoá kiểu [-1,1]
        x = (x.clamp(-1,1) + 1.0) * 127.5       # -> 0..255
    elif mx <= 1.5:                              # [0,1]
        x = x * 255.0
    x = x.permute(1,2,0).numpy()                 # HWC, RGB
    x = np.clip(x, 0, 255).astype("uint8")
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)    # BGR

def main():
    ap = argparse.ArgumentParser(description="Crop/alignment khuôn mặt về 160x160 bằng MTCNN")
    ap.add_argument("--input", required=True, help="File ảnh hoặc thư mục ảnh")
    ap.add_argument("--output", required=True, help="Thư mục xuất ảnh crop")
    ap.add_argument("--recursive", action="store_true", help="Duyệt đệ quy thư mục input")
    ap.add_argument("--min_conf", type=float, default=0.90, help="Ngưỡng xác suất detect tối thiểu")
    ap.add_argument("--image_size", type=int, default=160, help="Kích thước crop/alignment (FaceNet=160)")
    ap.add_argument("--post_process", action="store_true",
                    help="Giữ post_process=True (chuẩn hoá [-1,1]); mặc định tắt để lưu ảnh 0..255 dễ nhìn")
    ap.add_argument("--emit_list", type=str, default="",
                    help="Ghi danh sách đường dẫn ảnh crop ra file .txt để tiện đưa vào augmentation.py")
    ap.add_argument("--save_json", action="store_true",
                    help="Lưu metadata JSON (tên file gốc, conf, bbox)")
    ap.add_argument("--no_gpu", action="store_true", help="Buộc dùng CPU")
    args = ap.parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")
    mtcnn = MTCNN(keep_all=True,
                  device=device,
                  image_size=args.image_size,
                  margin=0,
                  post_process=args.post_process)   # mặc định False để lưu đẹp

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(in_path, args.recursive)
    print(f"[INFO] Found {len(images)} image(s).")

    saved_paths = []
    meta_all = []

    for p in images:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[SKIP] Cannot read {p}")
            continue
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        boxes, probs = mtcnn.detect(pil)
        if boxes is None:
            continue

        faces = mtcnn.extract(pil, boxes, save_path=None)
        if faces is None:
            continue

        for i, face in enumerate(faces):
            prob_i = None if probs is None else (None if probs[i] is None else float(probs[i]))
            if prob_i is not None and prob_i < args.min_conf:
                continue

            face_bgr = tensor_to_bgr_u8(face)
            out_name = f"{p.stem}_face{i+1}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), face_bgr)
            saved_paths.append(str(out_path))

            # (opt) lưu metadata đơn giản
            if args.save_json:
                b = list(map(float, boxes[i].tolist()))
                meta = {
                    "src": str(p),
                    "out": str(out_path),
                    "conf": prob_i,
                    "box": [b[0], b[1], b[2], b[3]],
                    "image_size": int(args.image_size)
                }
                meta_all.append(meta)

            print(f"[SAVE] {out_path}")

    # Ghi danh sách file cho augment
    if args.emit_list:
        Path(args.emit_list).parent.mkdir(parents=True, exist_ok=True)
        with open(args.emit_list, "w", encoding="utf-8") as f:
            f.write("\n".join(saved_paths))
        print(f"[INFO] Wrote list to: {args.emit_list}")

    if args.save_json and meta_all:
        meta_path = out_dir / "crop_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_all, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved crop metadata: {meta_path}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()