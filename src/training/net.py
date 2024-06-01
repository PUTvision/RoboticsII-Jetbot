from typing import List
import torch.nn as nn
import math

class ConvNet(nn.Module):
	def __init__(
		self, conv_sizes: List[int],kernel_sizes: List[int],max_pool_ks: List[int], linear_sizes: List[int], output_size: int, img_h:int,img_w:int
	):
		super(ConvNet, self).__init__()

		blocks = []

		out_formula = lambda x,k:math.floor(((x - k)/k) +1 )
		for a, b, k,mk in zip(conv_sizes[:-1], conv_sizes[1:],kernel_sizes,max_pool_ks):
			blocks.append(nn.Conv2d(a, b, kernel_size=k,padding="same"))
			# img_h = img_h - k +1
			# img_w = img_w - k +1
			blocks.append(nn.BatchNorm2d(b)) # batchnorm before relu
			blocks.append(nn.ReLU())
			#blocks.append(nn.Dropout2d(0.3)) # do not combine batch norm with dropout
			if mk != None:
				blocks.append(nn.MaxPool2d(mk))
				img_h = out_formula(img_h,mk)
				img_w = out_formula(img_w,mk)
		blocks.append(nn.Flatten())

		linear_sizes = [img_w*img_h*conv_sizes[-1]] + linear_sizes
		print(linear_sizes)
		for a, b in zip(linear_sizes[:-1], linear_sizes[1:]):
			print(a,b)
			blocks.append(nn.Linear(a, b))
			blocks.append(nn.ReLU())
			#blocks.append(nn.Dropout1d(0.3))

		blocks.append(nn.Linear(linear_sizes[-1], output_size))
		blocks.append(nn.Tanh())

		self.net = nn.Sequential(*blocks)

	def forward(self, x):
		return self.net(x)
