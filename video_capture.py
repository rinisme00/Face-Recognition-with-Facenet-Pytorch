# apps/crop_mtcnn.py
import argparse, os
from pathlib import Path
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def list_images(p: Path, recursive: bool):
    if p.is_file(): return [p]
    return [q for q in (p.rglob("*") if recursive else p.glob("*"))
            if q.suffix.lower() in IMG_EXTS]

def main():
    ap = argparse.ArgumentParser(description="Crop/alignment khuôn mặt về 160x160 bằng MTCNN")
    ap.add_argument("--input", required=True, help="File ảnh hoặc thư mục ảnh")
    ap.add_argument("--output", required=True, help="Thư mục xuất ảnh crop")
    ap.add_argument("--recursive", action="store_true", help="Duyệt đệ quy thư mục input")
    ap.add_argument("--min_conf", type=float, default=0.90, help="Ngưỡng xác suất detect tối thiểu")
    ap.add_argument("--image_size", type=int, default=160, help="Kích thước crop/alignment (FaceNet=160)")
    ap.add_argument("--no_gpu", action="store_true", help="Buộc dùng CPU")
    args = ap.parse_args()

    device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.no_gpu) else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device, image_size=args.image_size, margin=0)

    in_path = Path(args.input); out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(in_path, args.recursive)
    print(f"[INFO] Found {len(images)} image(s).")

    for p in images:
        # Đọc ảnh gốc (full-res), không resize
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[SKIP] Cannot read {p}"); continue
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # detect + extract (aligned faces [3,H,W] in [0..1] tensor)
        boxes, probs = mtcnn.detect(pil)
        if boxes is None: continue

        faces = mtcnn.extract(pil, boxes, save_path=None)
        if faces is None: continue

        # Lưu từng mặt về file .png 160x160 (hoặc .jpg nếu muốn)
        for i, face in enumerate(faces):
            if probs is not None and probs[i] is not None and probs[i] < args.min_conf:
                continue
            # face: Tensor [3,h,w] -> numpy HWC BGR để cv2.imwrite
            face_np = (face.permute(1,2,0).clamp(0,1).numpy()*255).astype("uint8")
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            out_name = f"{p.stem}_face{i+1}.png"
            cv2.imwrite(str(out_dir / out_name), face_bgr)
            print(f"[SAVE] {out_dir/out_name}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
