import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torchsummary import summary

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from typing import Tuple

import cv2
import numpy as np

from loss import WeightedMSELoss
from net import ConvNet
from data import JetbotDataset
from transforms import RandomHorizontalAndLabelFlip, HalfCrop

DATA_PATH = "./data/dataset"
BATCH_SIZE = 256
IMG_SIZE = 224
CHANNELS = 3


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    train_loss = 0.0

    for X, y in tqdm(train_loader, "batch"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def val_epoch(
    model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module, device: torch.device
):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(val_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs=10,
	patience=5,
):
	history = {"train_loss": [], "val_loss": []}

	prev_loss = 99999999999
	early_stopping = 0
	for epoch in range(epochs):
		train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
		val_loss = val_epoch(model, val_loader, loss_fn, device)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)

		print(
			f"\nEpoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n"
		)
		print("-" * 100)
		if val_loss < prev_loss:
			early_stopping = 0
			prev_loss = val_loss

		early_stopping += 1

		if early_stopping >= patience:
			print("No significant improvement")
			return history

	return history


def test(
    model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module, device: torch.device
):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(test_loader, "test_batch"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
    return test_loss / len(test_loader)


def display_img(img):
	cv2.imshow("in", cv2.cvtColor(np.transpose(img.numpy(),(1,2,0)),cv2.COLOR_BGR2RGB))
	cv2.waitKey(0)


def get_data(generator: torch.Generator) -> Tuple[DataLoader, DataLoader]:
	transform = transforms.Compose(
		[
			transforms.RandomRotation([5, 5]),
			transforms.RandomResizedCrop(IMG_SIZE, scale=(0.95, 1.0)),
			transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
			#transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
			#transforms.RandomHorizontalFlip(), #should not be here since the model would confuse left and right
			RandomHorizontalAndLabelFlip(0.5), # flips the image and the label
			#transforms.Grayscale(),
			HalfCrop(IMG_SIZE),
			transforms.ToDtype(torch.float32, scale=True),
		]
	)

	ds = JetbotDataset(
		DATA_PATH, transform, shift=5
	)  # predict the move 5 frames later and 10 framers later

	# for i in [181,2137,4312]:
	# 	img,lab = ds[i]
	# 	display_img(img)

	train_set, test_set = random_split(ds, [0.8, 0.2], generator=generator)

	return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
		test_set, batch_size=BATCH_SIZE, shuffle=False
	)


if __name__ == "__main__":
	if torch.backends.mps.is_available():
		device = torch.device("mps")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	print("Device is",device)

	generator = torch.Generator().manual_seed(42)
	train_loader, test_loader = get_data(generator)
	model = ConvNet(
		[CHANNELS, 24, 36, 48, 64,64],
		[5,5,5,3,3,3],
		[2,2,2,2,None,None],
		[100,50,10], 
		2,int(IMG_SIZE/2),IMG_SIZE)
	model.to(device)
	loss_fn = nn.L1Loss()#WeightedMSELoss(weights=torch.tensor([2,10]*3,dtype=torch.float32).to(device))#nn.L1Loss() # output is between -1 and 1, so when the difference is smaller than 1 the MSE actually makes it smaller
	optimizer = optim.SGD(model.parameters(), lr=0.001)
	summary(model, (CHANNELS, IMG_SIZE, int(IMG_SIZE/2)))

	history = train(
		model, train_loader, test_loader, loss_fn, optimizer, device, epochs=15
	)

	test_loss = test(model, test_loader, loss_fn, device)


	print(f"Test Loss: {test_loss:.4f}")

	torch_input = torch.randn(*(1,CHANNELS,IMG_SIZE,int(IMG_SIZE/2))).to(device)

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
