'''
Inception in chapter3 is used for hyper-parameters comparison experiments.
Inception Normal block:
					   x(c channels)
	       |           |           |           |
	    1x1 c/4     1x1 c/4     1x1 c/4     1x1 c/4
		branch1     branch2     branch3     branch4
	       |           |           |           |
	                       concat
					1x1 conv(c channels)
Inception Reduce block:
				x(c channels)
			|                   |
		  1x1 c                1x1 c
	branch1(stride=2)       branch2(stride=2)
			|                   |
					concat
			1x1 conv(2c channels)
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# normal 1x1
class Branch_0(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_0, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(),
		)
	
	def forward(self, x):
		return self.conv(x)


# normal 1x1 --> 3x1
class Branch_1(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_1, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU(),
			
			nn.Conv1d(out_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False, groups=out_planes),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU()
		)
	
	def forward(self, x):
		return self.conv(x)


# normal 1x1 --> 5x1
class Branch_2(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_2, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU(),
			
			nn.Conv1d(out_planes, out_planes, kernel_size=5, padding=2, stride=stride, bias=False, groups=out_planes),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU()
		)
	
	def forward(self, x):
		return self.conv(x)


# normal 1x1 --> 7x1
class Branch_3(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_3, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU(),
			
			nn.Conv1d(out_planes, out_planes, kernel_size=7, padding=3, stride=stride, bias=False, groups=out_planes),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU()
		)
	
	def forward(self, x):
		return self.conv(x)


# normal 1x1 --> 9x1
class Branch_4(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_4, self).__init__()
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU(),
			
			nn.Conv1d(out_planes, out_planes, kernel_size=9, padding=4, stride=stride, bias=False, groups=out_planes),
			nn.BatchNorm1d(out_planes, affine=True),
			nn.ReLU()
		)
	
	def forward(self, x):
		return self.conv(x)


# reduce 1x1 --> avgpool
class Branch_5(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_5, self).__init__()
		assert stride == 1 or 2
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(),
			
			nn.AvgPool1d(3, 2, padding=1)
		)
	
	def forward(self, x):
		x = self.conv(x)
		return x


# reduce 1x1 --> maxpool
class Branch_6(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Branch_6, self).__init__()
		assert stride == 1 or 2
		
		self.conv = nn.Sequential(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(),
			
			nn.MaxPool1d(3, 2, padding=1)
		)
	
	def forward(self, x):
		x = self.conv(x)
		return x


confg_list = {
	0: {'normal': [0, 1, 2, 3], 'reduce': [1, 5]},
	1: {'normal': [0, 1, 2, 4], 'reduce': [1, 5]},
	2: {'normal': [0, 1, 3, 4], 'reduce': [1, 5]},
	3: {'normal': [0, 2, 3, 4], 'reduce': [1, 5]}
}
branches_list = [Branch_0, Branch_1, Branch_2, Branch_3, Branch_4, Branch_5]


class InceptionModel(nn.Module):
	def __init__(self, mode=0, classes=4, num_blocks=4, out_filters=16, dropout=0.5):
		super(InceptionModel, self).__init__()
		self.num_blocks = num_blocks
		self.out_filters = out_filters
		self.dropout = dropout
		
		self.stem_conv = nn.Sequential(
			nn.Conv1d(1, self.out_filters, kernel_size=5, padding=2, stride=1, bias=False),
			nn.BatchNorm1d(self.out_filters),
			nn.ReLU(),
		)
		
		self.normal_ids, self.reduce_ids = confg_list[mode]['normal'], confg_list[mode]['reduce']
		self.normal_list = nn.ModuleList([])
		self.reduce_list = nn.ModuleList([])
		self.conv_list = nn.ModuleList([])
		for i in range(num_blocks):
			normals = nn.ModuleList([])
			for id in self.normal_ids:
				normals.append(branches_list[id](out_filters, out_filters // len(self.normal_ids), stride=1))
			self.normal_list.append(normals)
			self.conv_list.append(nn.Sequential(
				nn.Conv1d(out_filters // len(self.normal_ids) * len(self.normal_ids), out_filters, kernel_size=1,
				          padding=0, stride=1, bias=False),
				nn.BatchNorm1d(out_filters),
				nn.ReLU()
			))
			
			reduces = nn.ModuleList([])
			for id in self.reduce_ids:
				reduces.append(branches_list[id](out_filters, out_filters * 2 // len(self.reduce_ids), stride=2))
			self.reduce_list.append(reduces)
			self.conv_list.append(nn.Sequential(
				nn.Conv1d(out_filters * 2, out_filters * 2, kernel_size=1, padding=0, stride=1, bias=False),
				nn.BatchNorm1d(out_filters * 2),
				nn.ReLU()
			))
			out_filters *= 2
		
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
		self.dropout = nn.Dropout(self.dropout)
		self.classify = nn.Linear(out_filters, classes)
		
		# initialize
		self._initialize_weights()
	
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	
	def forward(self, x):
		x = self.stem_conv(x)
		for id in range(self.num_blocks):
			# normal
			temp = []
			for j in range(len(self.normal_ids)):
				temp.append(self.normal_list[id][j](x))
			x = torch.cat(temp, dim=1)
			x = self.conv_list[id * 2](x)
			
			# reduce
			temp = []
			for j in range(len(self.reduce_ids)):
				temp.append(self.reduce_list[id][j](x))
			x = torch.cat(temp, dim=1)
			x = self.conv_list[id * 2 + 1](x)
		
		x = self.global_avg_pool(x)
		x = self.dropout(x)
		features = x.view(x.shape[0], -1)
		out = self.classify(features)
		
		return out, features


class ReverseLayerF(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)
	
	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha
		return output, None
	

d_list = [1, 5, 7]
class InceptionTransfer(InceptionModel):
	def __init__(self, ff=False, dann=False, stem_kernel_size=5, mode=0, classes=4, num_blocks=4, out_filters=18,
	             dropout=0.5):
		super().__init__(mode=mode, classes=classes, num_blocks=num_blocks, out_filters=out_filters,
		                 dropout=dropout)
		self.ff = ff
		if ff:
			conv1 = nn.Conv1d(1, out_filters // 3, kernel_size=stem_kernel_size, stride=1, dilation=d_list[0],
			                  padding=int(d_list[0] * (stem_kernel_size - 1) // 2), bias=False)
			conv2 = nn.Conv1d(1, out_filters // 3, kernel_size=stem_kernel_size, stride=1, dilation=d_list[1],
			                  padding=int(d_list[1] * (stem_kernel_size - 1) // 2), bias=False)
			conv3 = nn.Conv1d(1, out_filters // 3, kernel_size=stem_kernel_size, stride=1, dilation=d_list[2],
			                  padding=int(d_list[2] * (stem_kernel_size - 1) // 2), bias=False)
			self.stem_conv = nn.ModuleList([conv1, conv2, conv3])
			self.bn = nn.BatchNorm1d(out_filters, affine=True)
			self.act = nn.Tanh()
		self.dann = dann
		if dann:
			self.domain_classifier = nn.Linear(out_filters * 2 ** num_blocks, 1)
			self.dann_act = nn.Sigmoid()
	
	def forward(self, x, alpha=1.):
		if self.ff:  # FFCNN
			x_list = []
			for conv in self.stem_conv:
				x_list.append(conv(x))
			x = torch.cat(x_list, dim=1)
			x = self.bn(x)
		# x = self.act(x)
		else:
			x = self.stem_conv(x)
		
		for id in range(self.num_blocks):
			# normal
			temp = []
			for j in range(len(self.normal_ids)):
				temp.append(self.normal_list[id][j](x))
			x = torch.cat(temp, dim=1)
			x = self.conv_list[id * 2](x)
			
			# reduce
			temp = []
			for j in range(len(self.reduce_ids)):
				temp.append(self.reduce_list[id][j](x))
			x = torch.cat(temp, dim=1)
			x = self.conv_list[id * 2 + 1](x)
		
		x = self.global_avg_pool(x)
		fc = x.view(x.shape[0], -1)
		if not self.dann:
			out = self.classify(fc)
			return out, fc
		else:
			out = self.classify(fc)
			fc = ReverseLayerF.apply(fc, alpha)
			domain_pred = self.domain_classifier(fc)
			domain_pred = self.dann_act(domain_pred)
			return out, fc


