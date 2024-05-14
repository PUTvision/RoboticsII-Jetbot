import os
import cv2
import timeit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print(os.getcwd())

from dataloading_utils import load_data, train_test_split, create_dataloader
from models import SimpleCNN

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, lr: float = 0.001, epochs: int = 10) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.epochs = epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.loss_history = []
        self.batch_inference_time = []
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for i, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                self.loss_history.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                
                if i % 10 == 0:
                    print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
                    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            
            mse = 0
            
            for data, labels in self.test_loader:
                # Calculate mse
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                
                loss = nn.functional.mse_loss(outputs, labels)
                mse += loss.item()
                
            # Calculate inference time
            self.measure_inference_time()
                
            mse /= len(self.test_loader)
            
            print(f'Mean Squared Error: {mse}')
            print(f'Average inference time: {self.average_inference_time * 1000:.3f} ms')
            
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        
    def measure_inference_time(self) -> None:
        random_data = torch.rand((1, 3, 224, 224)).to(self.device)
        # Timeit 1000 times
        self.average_inference_time = timeit.timeit(lambda: self.model(random_data), number=1000) / 1000
        


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset_path = 'dataset'
    debug = True
    batch_size = 32
    lr = 2e-4
    epochs = 3
    
    data, labels = load_data(dataset_path, debug=debug)
    
    # if debug:
    #     # Show the first image
    #     cv2.imshow('Image', data[0][0].transpose(1, 2, 0))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, debug=debug)    

    train_loader: DataLoader = create_dataloader(train_data, train_labels, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = create_dataloader(test_data, test_labels, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN()
    
    trainer = Trainer(model, train_loader, test_loader, lr=lr, epochs=epochs)
    trainer.train()
    trainer.test()

    trainer.save_model('models/model.pth')
    
    fig, ax = plt.subplots()
    plt.plot(trainer.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
    trainer.measure_inference_time()
    print(f'Average inference time: {trainer.average_inference_time}')
    
    
