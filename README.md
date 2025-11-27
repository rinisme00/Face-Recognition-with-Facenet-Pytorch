# Face-Recognition-with-Facenet-Pytorch
Real-time face detection and recognition with:

- **Detection**: MTCNN (from facenet-pytorch)
- **Embeddings**: InceptionResnetV1 (`pretrained on vggface2`)
- **Vector search**: FAISS - Facebook AI Similarity Search (cosine similarity on L2-normalized embeddings)
- **Interfaces**:
  - Command-line pipeline to build and query a face database
  - **Streamlit web app** (image upload + webcam via browser)

Works on **GPU (CUDA)** or **CPU**.
Recommended Python: **3.10.x** (tested with 3.10.18)

# 0. Cloning this repository
```
git clone https://github.com/rinisme00/Face-Recognition-with-Facenet-Pytorch.git
cd Face-Recognition-with-Facenet-Pytorch
```

# 1. Create the environment
## 1.1. Create & activate env (Python 3.10):
```
conda create -n <env_name> python=3.10 -y
conda activate <env_name>
```
## 1.2. PyTorch installation
Pick one install command that matches your system.

CUDA 12.1:
```
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

CUDA 11.8:
```
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

CPU only:
```
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

## 1.3. pip install -r requirements.txt

# 2. Building the FAISS vector database
## 2.1. Basic command:
```
python facenet_pipeline.py build-index --dataset-dir <your_faces_path>  --output-dir FaceNet_VectorDB
```
1. Loads MTCNN and FaceNet on GPU if available, otherwise CPU.

2. Scans `dataset-dir` and collects all images + labels.

3. For each image:
- Applies EXIF orientation
- Runs MTCNN to detect a face
- If detection probability ≥ detection threshold
- Feeds the aligned face into FaceNet to get a 512-dim embedding
- L2-normalizes the embedding

4. Builds a FAISS `IndexFlatIP` (inner product index → cosine similarity)

5. Saves to output-dir:
`employee_images.index`
`label_map.npy`
`embeddings.npy`

## 2.2. Optional config:
- --det-conf – detection confidence threshold for MTCNN
Default: value from `config.DEFAULT_DETECTION_CONF_THRESH`
Example:
```
python facenet_pipeline.py build-index --dataset-dir <your_faces_path> --output-dir FaceNet_VectorDB --det-conf 0.90
```

# 3. Querying the database from CLI
Once the index is built, you can query with another image:
```
python facenet_pipeline.py query index-dir FaceNet_VectorDBimage <path to query.jpg> top-k 5
```

Example:
```
python facenet_pipeline.py query \
    --index-dir FaceNet_VectorDB \
    --image test_images/alice_test.jpg \
    --top-k 5 \
    --recog-thresh 0.65
```