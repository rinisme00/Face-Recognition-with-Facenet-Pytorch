import glob

import os, pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".torch"                    # ví dụ: D:\Intro_to_DL\FaceRecognition\.torch

os.environ["TORCH_HOME"] = str(CACHE_DIR)         # phải set TRƯỚC khi import facenet_pytorch/torchvision
os.makedirs(CACHE_DIR / "checkpoints", exist_ok=True)

import torch
torch.hub.set_dir(str(CACHE_DIR))  # đảm bảo mọi thứ dùng đúng cache mới
print("TORCH_HOME:", torch.hub.get_dir())

import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
import numpy as np

IMG_PATH = './data/test_images'
DATA_PATH = './data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)
    
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        # print(usr)
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            # print('smt')
            embeds.append(model(trans(img).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 30 anh, kich thuoc [1,512]
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    # print(embedding)
    names.append(usr)
    
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)

if device == 'cpu':
    torch.save(embeddings, DATA_PATH+"/faceslistCPU.pth")
else:
    torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))