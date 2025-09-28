# apps/augmentation.py
import argparse
from pathlib import Path
import random
import cv2
import numpy as np

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def list_images(input_path: Path, recursive: bool):
    if input_path.is_file():
        return [input_path]
    it = input_path.rglob("*") if recursive else input_path.glob("*")
    return [p for p in it if p.suffix.lower() in IMG_EXTS]

# ------------ Deterministic ops ------------
def apply_brightness_contrast(img, alpha=1.0, beta=0.0):
    # alpha: nhân độ tương phản (1.0 giữ nguyên), beta: bù sáng (-255..255)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_gamma(img, gamma=1.0):
    if gamma <= 0: return img
    inv = 1.0 / gamma
    table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_gaussian_blur(img, ksize=0):
    if not ksize or ksize < 3 or ksize % 2 == 0:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_gaussian_noise(img, std=0.0, seed=None):
    if std <= 0: return img
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def simulate_jpeg(img, quality=100):
    quality = int(max(10, min(100, quality)))
    if quality >= 100:
        return img
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok: return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img

def build_suffix(alpha, beta, gamma, blur, noise, jpegq):
    parts = []
    if alpha != 1.0 or beta != 0.0: parts.append(f"bc{alpha:.2f}_{int(beta)}")
    if gamma != 1.0: parts.append(f"g{gamma:.2f}")
    if blur and blur >= 3 and blur % 2 == 1: parts.append(f"bl{blur}")
    if noise and noise > 0: parts.append(f"n{int(noise)}")
    if jpegq and jpegq < 100: parts.append(f"jq{jpegq}")
    return "__" + "_".join(parts) if parts else "__aug"

# ------------ Random helpers ------------
def parse_range2(s, typ=float):
    # "a b" -> (a,b)
    a, b = s.split()
    return typ(a), typ(b)

def rand_from_range(rng):
    lo, hi = rng
    return random.uniform(lo, hi)

def rand_blur_choice(s):
    # e.g. "0,3,5" -> [0,3,5], only odd accepted (0 = off)
    vals = [int(x.strip()) for x in s.split(",")]
    vals = [v for v in vals if v == 0 or (v >= 3 and v % 2 == 1)]
    return random.choice(vals) if vals else 0

def main():
    ap = argparse.ArgumentParser(description="Augment ảnh khuôn mặt: brightness/contrast, gamma, blur, noise, JPEG.")
    ap.add_argument("--input", help="Ảnh hoặc thư mục ảnh (bỏ qua nếu dùng --from_list)")
    ap.add_argument("--from_list", help="File .txt liệt kê các ảnh (mỗi dòng 1 path) - tiện từ crop_mtcnn.py --emit_list")
    ap.add_argument("--output", required=True, help="Thư mục xuất ảnh augment")
    ap.add_argument("--recursive", action="store_true", help="Duyệt đệ quy thư mục input")

    # Deterministic (mặc định)
    ap.add_argument("--alpha", type=float, default=1.0, help="Contrast gain (1.0=giữ nguyên)")
    ap.add_argument("--beta",  type=float, default=0.0, help="Brightness shift [-255..255]")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction (>0)")
    ap.add_argument("--blur_ksize", type=int, default=0, help="Gaussian blur kernel (số lẻ>=3), 0=tắt")
    ap.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise σ (0..50 gợi ý)")
    ap.add_argument("--jpeg_quality", type=int, default=100, help="Mô phỏng JPEG quality (10..100), 100=tắt")

    # Random mode: tạo N biến thể mỗi ảnh (ghi đè deterministic nếu dùng)
    ap.add_argument("--random", type=int, default=0, help="Nếu >0: tạo N biến thể/ảnh ngẫu nhiên trong khoảng")
    ap.add_argument("--rand_alpha", type=str, default="0.9 1.3", help="Khoảng alpha (contrast), vd: '0.9 1.3'")
    ap.add_argument("--rand_beta",  type=str, default="-20 20",  help="Khoảng beta (brightness), vd: '-20 20'")
    ap.add_argument("--rand_gamma", type=str, default="0.8 1.4", help="Khoảng gamma, vd: '0.8 1.4'")
    ap.add_argument("--rand_blur_choices", type=str, default="0,3,5", help="Tập blur kernel chọn ngẫu nhiên (0,3,5,7...)")
    ap.add_argument("--rand_noise", type=str, default="0 12",   help="Khoảng noise σ, vd: '0 12'")
    ap.add_argument("--rand_jpeg",  type=str, default="80 100", help="Khoảng JPEG quality, vd: '80 100'")

    ap.add_argument("--seed", type=int, default=None, help="Seed ngẫu nhiên (tùy chọn)")

    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    if args.from_list:
        txt = Path(args.from_list)
        if not txt.exists(): raise FileNotFoundError(txt)
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                if p.suffix.lower() in IMG_EXTS and p.exists():
                    files.append(p)
    elif args.input:
        files = list_images(Path(args.input), args.recursive)
    else:
        raise SystemExit("Cần --input hoặc --from_list")

    if not files:
        print("[WARN] Không tìm thấy ảnh đầu vào.")
        return

    print(f"[INFO] Found {len(files)} image(s). Output -> {out_dir}")

    # Nếu random mode
    if args.random and args.random > 0:
        r_alpha = parse_range2(args.rand_alpha, float)
        r_beta  = parse_range2(args.rand_beta,  float)
        r_gamma = parse_range2(args.rand_gamma, float)
        r_noise = parse_range2(args.rand_noise, float)
        r_jpeg  = parse_range2(args.rand_jpeg,  int)

        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[SKIP] Cannot read {p}")
                continue
            for k in range(args.random):
                a   = rand_from_range(r_alpha)
                b   = rand_from_range(r_beta)
                g   = max(0.05, rand_from_range(r_gamma))
                bl  = rand_blur_choice(args.rand_blur_choices)
                ns  = max(0.0, rand_from_range(r_noise))
                jq  = int(max(10, min(100, rand_from_range(r_jpeg))))

                out = apply_brightness_contrast(img, alpha=a, beta=b)
                out = apply_gamma(out, gamma=g)
                out = apply_gaussian_blur(out, ksize=int(bl))
                out = apply_gaussian_noise(out, std=float(ns))
                out = simulate_jpeg(out, quality=jq)

                suffix = build_suffix(a, b, g, int(bl), float(ns), jq)
                out_name = f"{p.stem}{suffix}_r{k+1}{p.suffix}"
                cv2.imwrite(str(out_dir / out_name), out)
                print(f"[SAVE] {out_dir/out_name}")
        print("[INFO] Done.")
        return

    # Deterministic mode
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] Cannot read {p}")
            continue

        out = apply_brightness_contrast(img, alpha=args.alpha, beta=args.beta)
        out = apply_gamma(out, gamma=args.gamma)
        out = apply_gaussian_blur(out, ksize=args.blur_ksize)
        out = apply_gaussian_noise(out, std=args.noise_std)
        out = simulate_jpeg(out, quality=args.jpeg_quality)

        suffix = build_suffix(args.alpha, args.beta, args.gamma, args.blur_ksize, args.noise_std, args.jpeg_quality)
        out_name = f"{p.stem}{suffix}{p.suffix}"
        cv2.imwrite(str(out_dir / out_name), out)
        print(f"[SAVE] {out_dir/out_name}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()