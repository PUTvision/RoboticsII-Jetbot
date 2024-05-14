import torch
import torch.nn as nn
import os 
import pandas as pd
import cv2
import numpy as np

DEBUG = True
DATASET_PATH = 'dataset'
BATCH_SZIE = 32

def load_data(data_folder_path: str, debug: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''
    Load the data from the dataset folder.
    
    Args:  
    
    `data_folder_path`: The path to the dataset folder.
    `debug`: If True, print some debug information.
    '''
    filenames = list(filter(lambda x: 'csv' in x, os.listdir(data_folder_path)))
    X,Y = [], []
    
    for filename in filenames:
        targets = pd.read_csv(os.path.join(data_folder_path, filename))
        Y.append(targets.to_numpy()[:, 1:])
        
        # Load images from the corresponding folders
        folder = filename.removesuffix('.csv')
        images = []
        imgs_names = os.listdir(os.path.join(data_folder_path, folder))
        for img_name in imgs_names:
            img = cv2.imread(os.path.join(data_folder_path, folder, img_name))
            images.append(img)
            
        X.append(images)
        
    if debug:
        print(f'Number of distinct runs in data: {len(X)}')
        print(f'Dimension of the first run: {len(X[0])} images')
        print(f'Dimension of the first image: {X[0][0].shape}')
        print(f'Number of distinct runs in labels: {len(Y)}')
        print(f'Dimension of the first label: {Y[0][0].shape}')
        
    return X, Y

def train_test_split(data: list[np.ndarray], labels: list[np.ndarray], test_size: float = 0.2, debug: bool = True) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    '''
    Split the data into training and testing sets.
    
    Args:
    
    `data`: The data to split.
    `labels`: The labels to split.
    `test_size`: The proportion of the data to be used for testing.
    `debug`: If True, print some debug information.
    '''
    split_idx = int(len(data) * (1 - test_size))
    train_data, test_data = data[:split_idx], data[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    if debug:
        print(f'Training data size: {len(train_data)}')
        print(f'Testing data size: {len(test_data)}')
        
    return train_data, test_data, train_labels, test_labels

def create_dataloader(data: list[np.ndarray], labels: list[np.ndarray], batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
    '''
    Create a DataLoader object from the data and labels.
    
    Args:
    
    `data`: The data to load.
    `labels`: The labels to load.
    `batch_size`: The size of the batch.
    `shuffle`: If True, shuffle the data.
    '''
    x = torch.tensor(data)
    y = torch.tensor(labels)
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    

if __name__ == '__main__':
    data, labels = load_data(DATASET_PATH, debug=DEBUG)
    
    if DEBUG:
        # Show the first image
        cv2.imshow('Image', data[0][0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, debug=DEBUG)    

    train_loader = create_dataloader(train_data, train_labels, batch_size=BATCH_SZIE, shuffle=True)
    test_loader = create_dataloader(test_data, test_labels, batch_size=BATCH_SZIE, shuffle=False)
    