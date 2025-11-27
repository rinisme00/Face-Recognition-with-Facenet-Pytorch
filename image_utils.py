from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch

from config import device


def load_image_exif(path: str) -> Image.Image:
    """
    Load image from path and respect EXIF orientation.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def normalize_pil(img: Image.Image) -> Image.Image:
    """
    Apply EXIF to a PIL image (used for webcam/frame uploads).
    """
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def extract_feature_from_path(
    image_path: str,
    recognizer,
    detector,
    conf_thresh: float,
) -> Optional[np.ndarray]:
    """
    Path → PIL Image → MTCNN → FaceNet embedding (512-dim, L2-normalized).
    Returns None if no good face is detected.
    """
    img = load_image_exif(image_path)
    return extract_feature_from_pil(img, recognizer, detector, conf_thresh)


def extract_feature_from_pil(
    img: Image.Image,
    recognizer,
    detector,
    conf_thresh: float,
) -> Optional[np.ndarray]:
    """
    PIL image → MTCNN → FaceNet embedding (512-dim, L2-normalized).
    Returns None if no good face is detected.
    """
    img = normalize_pil(img)
    # detector with return_prob=True
    face, prob = detector(img, return_prob=True)

    if face is None:
        return None

    p = float(np.array(prob).max())
    if p < conf_thresh:
        return None

    if face.ndim == 3:
        face = face.unsqueeze(0)
    face = face.to(device)

    with torch.no_grad():
        embedding = recognizer(face)  # (1, 512)

    embedding = embedding / embedding.norm(dim=1, keepdim=True)
    return embedding[0].cpu().numpy()  # (512,)


def embed_face_from_pil_with_prob(
    img: Image.Image,
    recognizer,
    detector,
    conf_thresh: float,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Same as extract_feature_from_pil but also returns detection prob.
    """
    img = normalize_pil(img)
    face, prob = detector(img, return_prob=True)

    if face is None:
        return None, None

    p = float(np.array(prob).max())
    if p < conf_thresh:
        return None, p

    if face.ndim == 3:
        face = face.unsqueeze(0)
    face = face.to(device)

    with torch.no_grad():
        embedding = recognizer(face)

    embedding = embedding / embedding.norm(dim=1, keepdim=True)
    return embedding[0].cpu().numpy(), p