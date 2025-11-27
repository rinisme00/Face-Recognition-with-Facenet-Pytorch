# Face-Recognition-with-Facenet-Pytorch
Real-time face detection and recognition with:

Detection: MTCNN (from facenet-pytorch)

Embeddings: InceptionResnetV1 (pretrained on casia-webface)

Webcam: OpenCV

Works on GPU (CUDA) or CPU.
Recommended Python: 3.10.x (tested with 3.10.18)

# 0. Cloning into repository
Suppose that you want to use our code for demo:
```
git clone https://github.com/rinisme00/Face-Recognition-with-Facenet-Pytorch.git
```

# 1. Create the environment
Create & activate env (Python 3.10):
```
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
```

Install PyTorch (GPU build). Pick the CUDA build your driver supports.
Examples (choose one that fits your system):

CUDA 12.1:
```
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

CUDA 11.8:
```
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

Install the libraries

pip install -r requirements.txt

CPU only:
```
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

# 2. Usage:

Shows live webcam with red bounding boxes (and optional landmarks if your script enables them).
```
python face_detect.py
```
Press ESC to quit.

If you have multiple cameras, change the index in cv2.VideoCapture(0) to 1, 2, …

Capturing frames from webcam for facelist.
```
python face_capture.py
```
Randomly capture 50 images when webcam is enabled.

Update person/faces:
Reads all images from data/test_images/<person>/*.jpg|*.jpeg|*.png, computes embeddings via InceptionResnetV1, averages per person, and saves:

data/faceslist.pth (GPU run) or data/faceslistCPU.pth (CPU run)

data/usernames.npy (array of person names)
```
python update_faces.py
```

Loads the embeddings & names from data/, opens the webcam, detects faces, computes embeddings, then matches to the nearest known person.
```
python face_recognition.py
```
Press ESC to quit.
