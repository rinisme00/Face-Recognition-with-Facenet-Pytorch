import argparse
from typing import List

import numpy as np
from tqdm import tqdm

from config import (
    device,
    DEFAULT_DETECTION_CONF_THRESH,
)
from models import create_detector_single, create_recognizer
from dataset_utils import collect_dataset
from image_utils import extract_feature_from_path
from vector_db import build_faiss_index, save_vector_db, load_vector_db_from_dir, search_similar


def run_build_index(dataset_dir: str, output_dir: str, det_conf: float) -> None:
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading models...")
    detector = create_detector_single()
    recognizer = create_recognizer()

    print(f"[INFO] Collecting dataset from: {dataset_dir}")
    df = collect_dataset(dataset_dir)
    print(f"[INFO] Found {len(df)} images")

    embeddings: List[np.ndarray] = []
    labels: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding images"):
        path = row["image_path"]
        label = row["label"]

        vec = extract_feature_from_path(path, recognizer, detector, conf_thresh=det_conf)
        if vec is None:
            print(f"[WARN] Skipped (no confident face): {path}")
            continue

        embeddings.append(vec)
        labels.append(label)

    if not embeddings:
        raise RuntimeError("No embeddings produced. Check dataset or thresholds.")

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
    print(f"[INFO] Embeddings shape: {emb_arr.shape}")

    print("[INFO] Building FAISS index...")
    index = build_faiss_index(emb_arr)

    print(f"[INFO] Saving vector DB to: {output_dir}")
    save_vector_db(index, labels, emb_arr, output_dir)
    print("[INFO] Done.")


def run_query(
    index_dir: str,
    image_path: str,
    top_k: int,
    det_conf: float,
    recog_thresh: float,
) -> None:
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading models...")
    detector = create_detector_single()
    recognizer = create_recognizer()

    print(f"[INFO] Loading vector DB from: {index_dir}")
    index, label_map = load_vector_db_from_dir(index_dir)

    print(f"[INFO] Extracting embedding for: {image_path}")
    vec = extract_feature_from_path(image_path, recognizer, detector, conf_thresh=det_conf)
    if vec is None:
        print("[ERROR] No confident face detected in query image.")
        return

    print(f"[INFO] Searching top-{top_k}...")
    results = search_similar(vec, index, label_map, k=top_k, recog_thresh=recog_thresh)

    print("\n=== Top-k results ===")
    for rank, r in enumerate(results, start=1):
        print(
            f"#{rank}: idx={r['index']}, "
            f"label={r['label']} (base={r['base_label']}), "
            f"sim={r['similarity']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="FaceNet + MTCNN + FAISS pipeline (build index & query)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- build-index
    p_build = sub.add_parser("build-index", help="Scan dataset and build FAISS index.")
    p_build.add_argument("--dataset-dir", required=True, type=str)
    p_build.add_argument("--output-dir", required=True, type=str)
    p_build.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DETECTION_CONF_THRESH,
        help=f"Detection threshold (default: {DEFAULT_DETECTION_CONF_THRESH})",
    )

    # ---- query
    p_query = sub.add_parser("query", help="Run one query image.")
    p_query.add_argument("--index-dir", required=True, type=str)
    p_query.add_argument("--image", required=True, type=str)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DETECTION_CONF_THRESH,
        help=f"Detection threshold (default: {DEFAULT_DETECTION_CONF_THRESH})",
    )
    p_query.add_argument(
        "--recog-thresh",
        type=float,
        default=0.65,
        help="Recognition threshold (cosine similarity).",
    )

    args = parser.parse_args()

    if args.command == "build-index":
        run_build_index(args.dataset_dir, args.output_dir, args.det_conf)
    elif args.command == "query":
        run_query(
            args.index_dir,
            args.image,
            args.top_k,
            args.det_conf,
            args.recog_thresh,
        )


if __name__ == "__main__":
    main()