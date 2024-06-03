import os
import pandas as pd
import torch
import torchvision

from typing import Callable, List, Optional, Tuple


class JetbotDataset(torchvision.datasets.VisionDataset):
	def __init__(
		self,
		path: str,
		preprocess: Optional[Callable] = None,
		augmentation: Optional[Callable] = None,
		shift=0,  # to adjust for latency in model
		weighted = False,
		min_w = 10,
		max_w = 30,
	):
		super(JetbotDataset, self).__init__(root=path, transforms=augmentation)
		self.labels, self.images, self.weights = load_files(path,max_w=max_w,min_w=min_w)
		self.shift = shift
		self.weighted = weighted
		self.preprocess = preprocess
		self.train_mode = False

	def __len__(self):
		return len(self.images)
	
	def train(self):
		self.train_mode = True
	
	def eval(self):
		self.train_mode = False

	def __getitem__(self, idx):
		img = torchvision.io.read_image(self.images[idx])
		#print(self.images[idx])
		label = torch.tensor(
			(
				self.labels[min(idx, len(self.labels) - 1)]
				# + self.labels[min(idx + self.shift, len(self.labels) - 1)]
				# + self.labels[min(idx + 2 * self.shift, len(self.labels) - 1)]
			),
			dtype=torch.float32,
		)
		weights = torch.empty(1)

		if self.weighted:
			weights = torch.tensor(self.weights[min(idx, len(self.weights) - 1)],dtype=torch.float32)

		if self.transforms != None and self.train_mode: # data augmentation
			img,label = self.transforms(img,label)

		if self.preprocess != None:
			img,label = self.preprocess(img,label)

		#return self.transforms(img, label) if self.weighted else self.transforms(img,label) + weights
		return img,label,weights


def load_files(path: str,min_w=10,max_w=20) -> Tuple[List[List[float]], List[str]]:
	labels = []
	weights = []
	images = []

	subdirs = [
		subdir for subdir in os.listdir(path) if os.path.isdir(f"{path}/{subdir}")
	]

	for subdir in subdirs:
		subdir_labels = load_labels(f"{path}/{subdir}.csv")

		for label in subdir_labels:
			frame, forward, right = label
			# if abs(forward) < 0.5 and abs(right) <0.3: # to stop the jetbot from standing
			# 	forward = 0.7

			labels.append([forward, right])
			weights.append(((max_w-min_w)*int(right!=0))+min_w)#abs(right))+min_w)

			img_name = str(int(label[0]))
			img_name = "0" * (4 - len(img_name)) + img_name

			images.append(f"{path}/{subdir}/{img_name}.jpg")

	return labels, images, weights


def load_labels(path: str) -> list[list[float]]:
    return pd.read_csv(path, header=None).values.tolist()
