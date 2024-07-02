import random
from functools import partial
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from lightning_fabric import seed_everything
from pydantic import BaseModel
from typing import List

from model.ImageTransforms import adjust_dpi, InfImageTransforms
from model.dataset import IAMDataset
from model.model import PadPool
import threading
import queue
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
import cv2 as cv
from InferencePipeline import InferencePipeline
from ModelLoader import ModelLoader
from model.utils import pickle_save

# ds = IAMDataset(root="/Users/tefannastasa/BachelorsWorkspace/BachModels/BachModels/data/raw/IAM", label_enc=None, parse_method="form" ,split="test")
# pickle_save(ds.label_enc, "./label_enc")

app = FastAPI()

model = ModelLoader()
pipeline = InferencePipeline(model)

class UrlList(BaseModel):
    urls: List[str]

transform = InfImageTransforms()
transform = transform.test_trnsf

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_GRAYSCALE)  # Read image and convert to grayscale
        cv.imwrite("./image.png", image)
        image = np.array(image)
        image = transform(image=image)["image"]
        image = torch.tensor(image)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch dimension
        print(image.size())
        return image
    else:
        raise HTTPException(status_code=400, detail=f"Failed to download image from {url}")


@app.post("/predict")
async def predict(url_list: UrlList, background_tasks: BackgroundTasks):
    results = []
    print("Prediction request received!")
    for url in url_list.urls:
        input_image = download_image(url)
        result_queue = pipeline.add_task(input_image)
        result = result_queue.get()
        result = "".join(result)
        results.append(result)

    print(" ".join(results))
    return {"prediction": " ".join(results)}

if __name__ == "__main__":
    import uvicorn
    seed_everything(5234)
    # for i in range(10):
    #   sel = random.randint(0, len(ds))
    #   image = ds[sel][0]
        # image = transform(image=image)["image"]
    #   image = torch.tensor(image, dtype=torch.float32)
    #   print(image.size())
    #   image = image.unsqueeze(0).unsqueeze(0)
    #   result_queue = pipeline.add_task(image)
    #   result = "".join(result_queue.get())
    #   print(result)
    SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "0.0.0.0")
    SERVER_PORT    = os.environ.get("SERVER_PORT", "27018")
    uvicorn.run(app, host=SERVER_ADDRESS, port=int(SERVER_PORT))
