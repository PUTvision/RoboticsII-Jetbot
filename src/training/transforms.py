import torch
from torchvision import transforms

class RandomHorizontalAndLabelFlip:
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image, label):
		#print("Before",label)
		if torch.rand(1) < self.p:
			image = transforms.functional.hflip(image)
			label[1] *= -1
			#print("Flipped",label)
		
		return image, label


class HalfCrop:
	def __init__(self, img_size):
		self.img_size = img_size
		self.half_size = int(img_size/2)

	def __call__(self, image, label):
		image = transforms.functional.crop(image,self.half_size,0,self.half_size,self.img_size)
		return image, label