import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import torchsummary

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from typing import Tuple

from src.training.net import ConvNet
from src.training.data import JetbotDataset

DATA_PATH = "./data/dataset"
GENERATOR = torch.Generator().manual_seed(42)
BATCH_SIZE = 64


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
):
    model.train()
    train_loss = 0.0

    for X, y in tqdm(train_loader):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def val_epoch(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(val_loader):
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
    epochs=10,
):
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = val_epoch(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"\nEpoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n"
        )
        print("-" * 100)

    return history


def test(model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
    return test_loss / len(test_loader)


def get_data() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.RandomRotation([10, 10]),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    ds = JetbotDataset(DATA_PATH, transform)

    train_set, test_set = random_split(ds, [0.8, 0.2], generator=GENERATOR)

    return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True), DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False
    )


if __name__ == "__main__":
    train_loader, test_loader = get_data()
    model = ConvNet([3, 8, 16, 32, 64], [64 * 10 * 10, 64], 2)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    torchsummary.summary(model, (3, 224, 224))

    history = train(model, train_loader, test_loader, loss_fn, optimizer, epochs=10)

    test_loss = test(model, test_loader, loss_fn)

    print(f"Test Loss: {test_loss:.4f}")
