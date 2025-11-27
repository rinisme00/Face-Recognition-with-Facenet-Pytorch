from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

from config import (
    device,
    DEFAULT_INDEX_PATH,
    DEFAULT_LABEL_MAP_PATH,
    DEFAULT_DETECTION_CONF_THRESH,
    DEFAULT_RECOGNITION_THRESH,
)
from models import create_detector_single, create_recognizer
from image_utils import embed_face_from_pil_with_prob
from vector_db import load_vector_db, search_similar


# ---- cached loaders (Streamlit-specific) ----

@st.cache_resource
def get_models():
    detector = create_detector_single()
    recognizer = create_recognizer()
    return detector, recognizer


@st.cache_resource
def get_vector_db():
    index, label_map = load_vector_db(DEFAULT_INDEX_PATH, DEFAULT_LABEL_MAP_PATH)
    return index, label_map


def recognize_image(
    img: Image.Image,
    top_k: int = 5,
) -> Tuple[Optional[str], Optional[float], Optional[List[Dict]]]:
    detector, recognizer = get_models()
    index, label_map = get_vector_db()

    vec, prob = embed_face_from_pil_with_prob(
        img, recognizer, detector, conf_thresh=DEFAULT_DETECTION_CONF_THRESH
    )
    if vec is None:
        return None, prob, None

    results = search_similar(
        vec,
        index,
        label_map,
        k=top_k,
        recog_thresh=DEFAULT_RECOGNITION_THRESH,
    )

    best = results[0]
    return best["label"], best["similarity"], results


def main():
    st.set_page_config(
        page_title="Face Recognition - FaceNet + FAISS",
        page_icon="👤",
        layout="centered",
    )

    st.title("Face Recognition with FaceNet + FAISS")
    st.write(
        "Upload a face image or use your webcam. "
        "The app will search the FAISS database and show the closest matches."
    )

    st.sidebar.header("Status")
    st.sidebar.write(f"**Device:** {device}")
    st.sidebar.write(f"**Index path:** `{DEFAULT_INDEX_PATH}`")
    st.sidebar.write(f"**Label map:** `{DEFAULT_LABEL_MAP_PATH}`")
    st.sidebar.write(f"**Det threshold:** {DEFAULT_DETECTION_CONF_THRESH}")
    st.sidebar.write(f"**Recog threshold:** {DEFAULT_RECOGNITION_THRESH}")

    mode = st.radio(
        "Choose input source:",
        ["Upload image", "Use webcam"],
        horizontal=True,
    )

    if mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
            with st.spinner("Recognizing..."):
                try:
                    name, score, results = recognize_image(img)
                except FileNotFoundError as e:
                    st.error(str(e))
                    return

            if name is None:
                st.error("No face detected or detection confidence too low.")
            else:
                st.success(f"Prediction: **{name}** (similarity = {score:.3f})")
                if results:
                    st.subheader("Top matches")
                    for r in results:
                        st.write(
                            f"- `{r['label']}` "
                            f"(base: `{r['base_label']}`, sim: {r['similarity']:.3f})"
                        )

    else:  # webcam
        cam_img = st.camera_input("Take a photo with your webcam")
        if cam_img is not None:
            img = Image.open(cam_img).convert("RGB")
            st.image(img, caption="Captured from camera", use_column_width=True)
            with st.spinner("Recognizing..."):
                try:
                    name, score, results = recognize_image(img)
                except FileNotFoundError as e:
                    st.error(str(e))
                    return

            if name is None:
                st.error("No face detected or detection confidence too low.")
            else:
                st.success(f"Prediction: **{name}** (similarity = {score:.3f})")
                if results:
                    st.subheader("Top matches")
                    for r in results:
                        st.write(
                            f"- `{r['label']}` "
                            f"(base: `{r['base_label']}`, sim: {r['similarity']:.3f})"
                        )


if __name__ == "__main__":
    main()