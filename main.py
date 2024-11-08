import random
from dataclasses import dataclass
import os
import copy

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torchvision import transforms
import numpy as np


@dataclass
class TrainingArgs:
    c: float = 0.01
    lr: float = 0.1
    batchsize: int = 5
    epoch: int = 10
    device: str = "cpu"


Categories = ["bus", "motorcycle"]
flat_data_arr = []  # input array
target_arr = []  # output array
datadir = "Dataset/"
# path which contains all the categories of images

for i in Categories:
    print(f"loading... category : {i}")
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = Image.open(os.path.join(path, img))
        img_array = img_array.convert("RGB").resize((150, 150))
        to_tensor = transforms.ToTensor()
        img_resized = torch.flatten(to_tensor(img_array))
        flat_data_arr.append(img_resized)
        target_arr.append(Categories.index(i))
    print(f"loaded category:{i} successfully")


for i in range(len(target_arr)):
    if target_arr[i] == 0:
        target_arr[i] = 1
    else:
        target_arr[i] = -1

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

X_train = flat_data
y_train = target

dim = len(X_train[0])
w = torch.autograd.Variable(torch.rand(dim), requires_grad=True)
b = torch.autograd.Variable(torch.rand(1), requires_grad=True)

step_size = 1e-3
num_epochs = 500
minibatch_size = 20

for epoch in range(num_epochs):
    print(f"Running epoch {epoch}")
    inds = [i for i in range(len(X_train))]
    random.shuffle(inds)
    for i in range(len(inds)):
        L = (
            max(
                0,
                1
                - y_train[inds[i]] * (torch.dot(w, torch.Tensor(X_train[inds[i]])) - b),
            )
            ** 2
        )
        if (
            L != 0
        ):  # if the loss is zero, Pytorch leaves the variables as a float 0.0, so we can't call backward() on it
            L.backward()
            w.data -= step_size * w.grad.data  # step
            b.data -= step_size * b.grad.data  # step
            w.grad.data.zero_()
            b.grad.data.zero_()


print("plane equation:  w=", w.detach().numpy(), "b =", b.detach().numpy()[0])


def accuracy(X, y):
    correct = 0
    for i in range(len(y)):
        y_predicted = int(
            np.sign((torch.dot(w, torch.Tensor(X[i])) - b).detach().numpy()[0])
        )
        if y_predicted == y[i]:
            correct += 1
    return float(correct) / len(y)


print("train accuracy", accuracy(X_train, y_train))
