from facenet_pytorch import MTCNN, InceptionResnetV1
from config import device


def create_detector_single():
    """
    MTCNN for a single best face.
    """
    return MTCNN(
        image_size=160,
        margin=20,
        min_face_size=15,
        thresholds=[0.5, 0.6, 0.6],
        factor=0.709,
        keep_all=False,
        post_process=True,
        device=device,
    )


def create_detector_multi():
    """
    MTCNN for multiple faces (if you need it).
    """
    return MTCNN(
        image_size=160,
        margin=20,
        min_face_size=15,
        thresholds=[0.5, 0.6, 0.6],
        factor=0.709,
        keep_all=True,
        post_process=True,
        device=device,
    )


def create_recognizer():
    """
    InceptionResnetV1 backbone
    """
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model