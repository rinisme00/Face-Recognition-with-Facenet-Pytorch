import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from config import (
    device,
    DEFAULT_DETECTION_CONF_THRESH,
    DEFAULT_RECOGNITION_THRESH,
    DEFAULT_INDEX_PATH,
    DEFAULT_LABEL_MAP_PATH,
)
from models import create_detector_multi, create_recognizer
from image_utils import normalize_pil
from vector_db import load_vector_db, search_similar


# ================================
# Helpers: detection + embeddings
# ================================

def detect_and_embed_all_faces_pil(
    img_pil: Image.Image,
    detector,
    recognizer,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect all faces in a PIL image and compute embeddings.

    Returns:
        boxes:      (N, 4) array of [x1, y1, x2, y2] in image coords, or None
        probs:      (N,) array of detection probabilities, or None
        embeddings: (N, 512) array of L2-normalized FaceNet embeddings, or None
    """
    img_pil = normalize_pil(img_pil)  # EXIF + RGB

    # 1) Get boxes & detection probabilities
    boxes, probs = detector.detect(img_pil)  # boxes: (N, 4), probs: (N,)
    if boxes is None or probs is None:
        return None, None, None

    # 2) Get aligned face tensors for each detection
    faces = detector(img_pil)  # shape (N, 3, 160, 160) or None
    if faces is None:
        return None, None, None

    faces = faces.to(device)

    # 3) FaceNet embeddings
    with torch.no_grad():
        embs = recognizer(faces)  # (N, 512)

    # 4) L2-normalize so inner product == cosine similarity
    embs = embs / embs.norm(dim=1, keepdim=True)
    embs = embs.cpu().numpy()  # (N, 512)

    return boxes, probs, embs

# ================================
# Image mode processing
# ================================

def process_single_image(
    image_path: str,
    detector,
    recognizer,
    index,
    label_map,
    face_mode: str,
    det_conf: float,
    recog_thresh: float,
    top_k: int,
) -> bool:
    """
    Process ONE image path:
      - detect faces
      - search TOP-K in FAISS
      - draw bounding boxes + name + similarity
      - print TOP-K results to console

    Returns:
        True  -> continue to next image
        False -> user requested quit (q or ESC)
    """
    print(f"\n[IMAGE] {image_path}")
    if not os.path.exists(image_path):
        print(f"  [ERROR] File not found.")
        return True

    img_pil = Image.open(image_path).convert("RGB")
    boxes, probs, embs = detect_and_embed_all_faces_pil(img_pil, detector, recognizer)

    # Prepare BGR image for drawing
    frame_rgb = np.array(img_pil)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if boxes is None or probs is None or embs is None:
        print("  [WARN] No face detected.")
        window_name = f"Image - {os.path.basename(image_path)}"
        cv2.imshow(window_name, frame_bgr)
        print("  Press any key for next image, or 'q'/ESC to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord("q"):
            cv2.destroyWindow(window_name)
            return False
        cv2.destroyWindow(window_name)
        return True

    num_faces = len(boxes)
    print(f"  [INFO] Detected {num_faces} faces (before threshold).")

    # Decide which detections to keep based on single/multi mode
    if face_mode == "single":
        # Use only the best detection (highest prob)
        best_idx = int(np.argmax(probs))
        candidate_indices = [best_idx]
    else:
        # Use all detections
        candidate_indices = list(range(num_faces))

    # For each chosen face → FAISS search + draw box
    for i in candidate_indices:
        p = probs[i]
        if p is None or p < det_conf:
            continue

        emb = embs[i]
        results = search_similar(
            emb,
            index,
            label_map,
            k=top_k,
            recog_thresh=recog_thresh,
        )

        # Log TOP-K results
        print(f"  [Face {i}] det_prob={p:.3f}")
        for rank, r in enumerate(results, start=1):
            print(
                f"    #{rank}: idx={r['index']}, "
                f"label={r['label']} (base={r['base_label']}), "
                f"sim={r['similarity']:.4f}"
            )

        best = results[0]
        label = best["label"]
        sim = best["similarity"]

        # Draw bounding box + text
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{label} ({sim:.2f})"
        cv2.putText(
            frame_bgr,
            text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    window_name = f"Image - {os.path.basename(image_path)}"
    cv2.imshow(window_name, frame_bgr)
    print("  Press any key for next image, or 'q'/ESC to quit.")
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == ord("q"):
        cv2.destroyWindow(window_name)
        return False

    cv2.destroyWindow(window_name)
    return True


def run_image_mode(
    image_paths: List[str],
    index_path: str,
    label_map_path: str,
    face_mode: str,
    det_conf: float,
    recog_thresh: float,
    top_k: int,
) -> None:
    """
    Run recognition on a list of image paths.
    """
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading models (multi-face MTCNN + FaceNet)...")
    detector = create_detector_multi()
    recognizer = create_recognizer()

    print(f"[INFO] Loading FAISS index:\n  index = {index_path}\n  labels = {label_map_path}")
    index, label_map = load_vector_db(index_path, label_map_path)

    for path in image_paths:
        cont = process_single_image(
            image_path=path,
            detector=detector,
            recognizer=recognizer,
            index=index,
            label_map=label_map,
            face_mode=face_mode,
            det_conf=det_conf,
            recog_thresh=recog_thresh,
            top_k=top_k,
        )
        if cont is False:
            break

    cv2.destroyAllWindows()
    print("[INFO] Image mode finished.")
    

# ================================
# Livecam mode processing
# ================================

def process_frame_livecam(
    frame_bgr: np.ndarray,
    detector,
    recognizer,
    index,
    label_map,
    face_mode: str,
    det_conf: float,
    recog_thresh: float,
    top_k: int,
    frame_idx: int,
):
    """
    Recognize faces in one webcam frame.

    Returns:
        annotated_frame_bgr, log_lines(list of str)
    """
    # BGR -> RGB -> PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    boxes, probs, embs = detect_and_embed_all_faces_pil(img_pil, detector, recognizer)

    log_lines: List[str] = []

    if boxes is None or probs is None or embs is None:
        log_lines.append(f"[Frame {frame_idx}] No face detected.")
        return frame_bgr, log_lines

    num_faces = len(boxes)

    # Choose which faces to use (single vs multi)
    if face_mode == "single":
        best_idx = int(np.argmax(probs))
        candidate_indices = [best_idx]
    else:
        candidate_indices = list(range(num_faces))

    selected_boxes = []
    selected_labels = []
    selected_sims = []

    for i in candidate_indices:
        p = probs[i]
        if p is None or p < det_conf:
            continue

        emb = embs[i]
        results = search_similar(
            emb,
            index,
            label_map,
            k=top_k,
            recog_thresh=recog_thresh,
        )
        best = results[0]
        label = best["label"]
        sim = best["similarity"]
        base_label = best["base_label"]

        selected_boxes.append(boxes[i])
        selected_labels.append(label)
        selected_sims.append(sim)

        log_lines.append(
            f"[Frame {frame_idx}][Face {i}] {label} "
            f"(base={base_label}, sim={sim:.3f})"
        )

    if not selected_boxes:
        log_lines.append(
            f"[Frame {frame_idx}] No face above det_conf={det_conf:.2f}."
        )
        return frame_bgr, log_lines

    # Draw bounding boxes + labels on frame
    for box, label, sim in zip(selected_boxes, selected_labels, selected_sims):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{label} ({sim:.2f})"
        cv2.putText(
            frame_bgr,
            text,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return frame_bgr, log_lines


def run_livecam_mode(
    index_path: str,
    label_map_path: str,
    camera_id: int,
    face_mode: str,
    det_conf: float,
    recog_thresh: float,
    top_k: int,
) -> None:
    """
    Run recognition on live webcam frames.
    Logs every frame to console and shows bounding boxes + labels.
    """
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading models (multi-face MTCNN + FaceNet)...")
    detector = create_detector_multi()
    recognizer = create_recognizer()

    print(f"[INFO] Loading FAISS index:\n  index = {index_path}\n  labels = {label_map_path}")
    index, label_map = load_vector_db(index_path, label_map_path)

    print(f"[INFO] Opening camera id={camera_id} ...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Press 'q' or ESC to quit.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        frame_idx += 1

        frame_out, log_lines = process_frame_livecam(
            frame_bgr=frame,
            detector=detector,
            recognizer=recognizer,
            index=index,
            label_map=label_map,
            face_mode=face_mode,
            det_conf=det_conf,
            recog_thresh=recog_thresh,
            top_k=top_k,
            frame_idx=frame_idx,
        )

        # Log every frame's recognition info
        for line in log_lines:
            print(line)

        cv2.imshow("Livecam - Face Recognition", frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Livecam mode finished.")


# ================================
# CLI
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Test program for FaceNet + MTCNN + FAISS (image paths & livecam, single/multi-face)."
    )

    parser.add_argument(
        "--source",
        choices=["image", "livecam"],
        required=True,
        help="Input source: 'image' for image paths, 'livecam' for webcam stream.",
    )
    parser.add_argument(
        "--faces",
        choices=["single", "multi"],
        default="single",
        help="Face mode: 'single' main face or 'multi' all faces per frame/image.",
    )

    # Common FAISS paths
    parser.add_argument(
        "--index-path",
        type=str,
        default=DEFAULT_INDEX_PATH,
        help=f"Path to FAISS index file (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--label-map-path",
        type=str,
        default=DEFAULT_LABEL_MAP_PATH,
        help=f"Path to label map .npy (default: {DEFAULT_LABEL_MAP_PATH}).",
    )

    # Image mode
    parser.add_argument(
        "--image-paths",
        nargs="+",
        help="One or more image paths (required if --source=image).",
    )

    # Livecam mode
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )

    # Recognition settings
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k neighbors to search in FAISS (default: 5).",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DETECTION_CONF_THRESH,
        help=f"Detection confidence threshold (default: {DEFAULT_DETECTION_CONF_THRESH}).",
    )
    parser.add_argument(
        "--recog-thresh",
        type=float,
        default=DEFAULT_RECOGNITION_THRESH,
        help=f"Recognition similarity threshold (default: {DEFAULT_RECOGNITION_THRESH}).",
    )

    args = parser.parse_args()

    # Validate mode-specific args
    if args.source == "image" and not args.image_paths:
        parser.error("--image-paths is required when --source=image")

    if args.source == "image":
        run_image_mode(
            image_paths=args.image_paths,
            index_path=args.index_path,
            label_map_path=args.label_map_path,
            face_mode=args.faces,
            det_conf=args.det_conf,
            recog_thresh=args.recog_thresh,
            top_k=args.top_k,
        )
    else:  # livecam
        run_livecam_mode(
            index_path=args.index_path,
            label_map_path=args.label_map_path,
            camera_id=args.camera_id,
            face_mode=args.faces,
            det_conf=args.det_conf,
            recog_thresh=args.recog_thresh,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()