from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os

Categories = ["bus", "motorcycle", "truck", "car"]
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = "Dataset/"
# path which contains all the categories of images
for i in Categories:
    print(f"loading... category : {i}")
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        if img.endswith(".jpg") or img.endswith(".jpeg"):
            img_array = Image.open(os.path.join(path, img))
            img_array = img_array.convert("RGB").resize((150, 150))
            to_tensor = transforms.ToTensor()
            img_resized = torch.flatten(to_tensor(img_array))
            print(img_resized.shape)
            flat_data_arr.append(img_resized)
            target_arr.append(Categories.index(i))
    print(f"loaded category:{i} successfully")

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
