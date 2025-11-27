import os
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding size of FaceNet
VECTOR_DIM = 512

# Thresholds
DEFAULT_DETECTION_CONF_THRESH = float(os.getenv("DETECTION_CONF_THRESH", "0.9"))
DEFAULT_RECOGNITION_THRESH = float(os.getenv("RECOGNITION_THRESH", "0.65"))

# Paths for DB files
DEFAULT_INDEX_PATH = os.getenv(
    "FACENET_INDEX_PATH", "FaceNet_VectorDB/employee_images.index"
)
DEFAULT_LABEL_MAP_PATH = os.getenv(
    "FACENET_LABEL_MAP_PATH", "FaceNet_VectorDB/label_map.npy"
)
DEFAULT_EMBEDDINGS_PATH = os.getenv(
    "FACENET_EMBEDDINGS_PATH", "FaceNet_VectorDB/embeddings.npy"
)