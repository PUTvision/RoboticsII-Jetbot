import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        return (self.weights * (output - target) ** 2).mean()

class WeightedSpaceMSELoss(nn.Module):
	def __init__(self):
		super(WeightedSpaceMSELoss, self).__init__()

	def forward(self, output, target,weights):
		# weights = (torch.abs(target[:,1]*10)+2)
		weights = weights.reshape(weights.shape[0],1)
		return (weights * (output - target) ** 2).mean()