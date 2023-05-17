import torch
import torch.nn as nn


OPS = {
	'conv_3x3' : lambda C, stride, affine: Conv(C, C, 3, stride, 1, affine=affine),
	'conv_5x5' : lambda C, stride, affine: Conv(C, C, 5, stride, 2, affine=affine),
	'conv_7x7' : lambda C, stride, affine: Conv(C, C, 7, stride, 3, affine=affine),
	'conv_9x9' : lambda C, stride, affine: Conv(C, C, 9, stride, 4, affine=affine),
	'conv_11x11' : lambda C, stride, affine: Conv(C, C, 11, stride, 5, affine=affine),
	'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
	'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1),
	'max_pool_5x5' : lambda C, stride, affine: nn.MaxPool1d(5, stride=stride, padding=2),
	'avg_pool_5x5' : lambda C, stride, affine: nn.AvgPool1d(5, stride=stride, padding=2),
	'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
}


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	
	def forward(self, x):
		return x


class Conv(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
		super(Conv, self).__init__()
		self.op = nn.Sequential(
			nn.Conv1d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
			nn.BatchNorm1d(C_out, affine=affine),
			nn.ReLU(inplace=False),
		)
	
	def forward(self, x):
		return self.op(x)



class FactorizedReduce(nn.Module):
	def __init__(self, C_in, C_out, affine=True):
		super(FactorizedReduce, self).__init__()
		assert C_out % 2 == 0
		
		self.conv = nn.Conv1d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False)
		self.bn = nn.BatchNorm1d(C_out, affine=affine)
		self.relu = nn.ReLU(inplace=False)
	
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		out = self.relu(x)
		return out


