import sys
import os

import cv2 as cv

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchsummary import summary

from dataloader import JetbotDataset,SubsetJetbotDataset

from tqdm import tqdm

# class ConvolutionalModel(nn.Module):
#     def __init__(self,input_size=224):
#         super(ConvolutionalModel, self).__init__()

#         conv_params = [3,8,16,32,64]

#         self.convs = []
#         for a,b in zip(conv_params[:-1],conv_params[1:]):
#             self.convs.append(nn.Conv2d(a,b,kernel_size=5))

#         linear_params = [10*10*64,64,2]
#         self.linears = []
#         for a,b in zip(linear_params[:-1],linear_params[1:]):
#             self.linears.append(nn.Linear(a,b))
#         self.output = self.linears.pop(-1)

#     def forward(self,x):

#         for conv in self.convs:
#             x = F.relu(F.max_pool2d(conv(x),2))
#         x = x.view(-1,10*10*64)
#         for lin in self.linears:
#             x = F.relu(lin(x))
#         return F.tanh(x)


def create_model():
    layers = []
    conv_params = [3, 8, 16, 32, 64]
    for a, b in zip(conv_params[:-1], conv_params[1:]):
        layers.append(nn.Conv2d(a, b, kernel_size=5))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Flatten())
    linear_params = [10 * 10 * 64, 64, 2]
    out = linear_params.pop(-1)
    for a, b in zip(linear_params[:-1], linear_params[1:]):
        layers.append(nn.Linear(a, b))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(linear_params[-1], out))
    layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def training_loop(
    model,
    train_dataloader,
    val_dataloader,
    epochs=10,
    loss=nn.L1Loss,
    optimizer=optim.SGD,
    optimizer_params=[],
):
    criterion = loss()
    optimizer = optimizer(model.parameters(), *optimizer_params)
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for i, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i == 5:
                break
    train_loss = train_loss / len(train_dataloader)

    model.eval()
    val_loss = 0.0
    # with torch.no_grad():
    #     for inputs, labels in val_dataloader:
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item()
    # val_loss = val_loss / len(val_dataloader)

    print(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {0:.4f}"
    )


def main(*argv):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.5,0.5,0.5,0.5),
        transforms.GaussianBlur(5, sigma=(0.1, 2.0))
    ])
    model = create_model()
    summary(model, (3, 224, 224))

    path = "./data/dataset/"

    data = JetbotDataset(path,batch_size=64)

    idx = np.arange(len(data))

    train_idx, rest_idx = train_test_split(idx, test_size=0.3)
    test_idx, val_idx = train_test_split(rest_idx, test_size=0.5)

    train_data = SubsetJetbotDataset(path,train_idx,train_transform)
    val_data = SubsetJetbotDataset(path,val_idx,train_transform)

    # img,lab = train_data[182]
    # cv.imshow("in", cv.cvtColor(np.transpose(img.numpy(),(1,2,0)),cv.COLOR_BGR2RGB))
    # cv.waitKey(0)
    # return 0

    training_loop(model,DataLoader(train_data,64),DataLoader(val_data,64),epochs=1)

    torch_input = torch.randn(*(1,3,224,224))

    torch.onnx.export(
        model,                      # model to be exported
        torch_input,                # sample input tensor
        "./model.onnx",             # where to save the ONNX file
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=11,           # specify the ONNX opset version
        do_constant_folding=True,   # perform constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # dynamic axes for variable batch size
    )

    return 0


if __name__ == "__main__":
    main(*sys.argv)
