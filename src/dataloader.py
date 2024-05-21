import os
import numpy as np
import pandas as pd

import cv2 as cv

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image

class JetbotDataset(Dataset):
    def __init__(self,path,**kwargs):
        #super(CustomDataLoader,self).__init__(None,**kwargs)
        self.path = path
        self.labels_names ,self.images = self.load_files(path)
        self.folder_names = [label_name[:-4] for label_name in self.labels_names]
        self.labels = self.load_labels(path,self.labels_names)
        self.counts = tuple(map(len,self.images))
        self.length = sum(self.counts)

    def load_files(self,path):
        files = [files for subdir, dirs, files  in os.walk(path)]
        return files[0],files[1:]

    def load_labels(self,path,labels_names):
        return [pd.read_csv(path+file_name,header=None) for file_name in labels_names]

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        for folder,count in enumerate(self.counts):
            if idx - (count-1) <= 0:
                break
            idx -= (count-1)

        label = self.labels[folder].iloc[idx,1:] if len(self.labels[folder]) > idx else self.labels[folder].iloc[-1,1:]

        image = cv.imread(self.path+self.folder_names[folder]+"/"+self.images[folder][idx])#read_image(self.path+self.folder_names[folder]+"/"+self.images[folder][idx])

        return image,label.to_numpy()

    # def __iter__(self):
    #     for idx in range(len(self)):
    #         yield self[idx]

class SubsetJetbotDataset(JetbotDataset):
    def __init__(self,path,subset_ids,transforms=None,**kwargs):
        super(SubsetJetbotDataset,self).__init__(path)
        self.subset_ids = subset_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.subset_ids)
    
    def __getitem__(self,idx):
        img,label = super(SubsetJetbotDataset,self).__getitem__(self.subset_ids[idx])
        return (self.transforms(img) if self.transforms != None else img),label
