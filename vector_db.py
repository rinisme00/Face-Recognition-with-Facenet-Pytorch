import os
from typing import List, Dict, Tuple

import faiss
import numpy as np

from config import VECTOR_DIM, DEFAULT_RECOGNITION_THRESH

INDEX_FILENAME = "employee_images.index"
LABEL_MAP_FILENAME = "label_map.npy"
EMBEDDINGS_FILENAME = "embeddings.npy"


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an IndexFlatIP index for cosine similarity.
    Assumes embeddings are already L2-normalized.
    """
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(embeddings.astype(np.float32))
    return index


def save_vector_db(
    index: faiss.IndexFlatIP,
    label_map: List[str],
    embeddings: np.ndarray,
    output_dir: str,
    index_filename: str = INDEX_FILENAME,
    label_map_filename: str = LABEL_MAP_FILENAME,
    embeddings_filename: str = EMBEDDINGS_FILENAME,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, index_filename))
    np.save(os.path.join(output_dir, label_map_filename), np.array(label_map))
    np.save(os.path.join(output_dir, embeddings_filename), embeddings)


def load_vector_db_from_dir(
    directory: str,
    index_filename: str = INDEX_FILENAME,
    label_map_filename: str = LABEL_MAP_FILENAME,
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    index_path = os.path.join(directory, index_filename)
    label_map_path = os.path.join(directory, label_map_filename)
    return load_vector_db(index_path, label_map_path)


def load_vector_db(
    index_path: str,
    label_map_path: str,
) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")

    index = faiss.read_index(index_path)
    label_map = np.load(label_map_path, allow_pickle=True)
    return index, label_map


def search_similar(
    query_vector: np.ndarray,
    index: faiss.IndexFlatIP,
    label_map: np.ndarray,
    k: int = 5,
    recog_thresh: float = DEFAULT_RECOGNITION_THRESH,
) -> List[Dict]:
    """
    Returns list of:
      { "index", "label", "base_label", "similarity" }
    """
    q = np.expand_dims(query_vector.astype(np.float32), axis=0)
    sims, idxs = index.search(q, k)

    results: List[Dict] = []
    for r in range(len(idxs[0])):
        idx = int(idxs[0][r])
        sim = float(sims[0][r])
        base_label = label_map[idx]
        label = base_label if sim >= recog_thresh else "Unknown"

        results.append(
            {
                "index": idx,
                "label": label,
                "base_label": base_label,
                "similarity": sim,
            }
        )

    return results