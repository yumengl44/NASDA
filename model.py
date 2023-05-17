import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Inception_operations import *
from genotypes import *
from utils import MMD_loss, count_parameters_in_MB


class Branch(nn.Module):
	def __init__(self, C, stride, name):
		super(Branch, self).__init__()
		self.op = OPS[name](C, stride, True)
	
	def forward(self, x):
		return self.op(x)


class Block(nn.Module):
	def __init__(self, C_in, C_out, reduction, genotypes):
		super(Block, self).__init__()
		self.reduction = reduction
		
		self.conv_list = nn.ModuleList()
		self.branch_list = nn.ModuleList()
		self.num_branch = 2 if reduction else 4
		if reduction:
			op_names = genotypes.reduce
		else:
			op_names = genotypes.normal
		
		for i in range(self.num_branch):
			self.conv_list.append(
				nn.Sequential(
					nn.Conv1d(C_in, C_out // self.num_branch, kernel_size=1, stride=1, bias=False),
					nn.BatchNorm1d(C_out // self.num_branch, affine=True),
					nn.ReLU()
				)
			)
			stride = 2 if reduction else 1
			self.branch_list.append(Branch(C_out // self.num_branch, stride, op_names[i]))
		
		self.conv = nn.Sequential(
			nn.Conv1d(C_out // self.num_branch * self.num_branch, C_out, kernel_size=1, padding=0, stride=1,
			          bias=False),
			nn.BatchNorm1d(C_out),
			nn.ReLU()
		)
	
	def forward(self, x):
		temp = []
		for i in range(self.num_branch):
			s = self.conv_list[i](x)
			s = self.branch_list[i](s)
			temp.append(s)
		s = torch.cat(temp, dim=1)
		out = self.conv(s)
		
		return out


class Inception(nn.Module):
	def __init__(self, genotypes, criterion=None, classes=4, num_blocks=8, C=16, num_task=3,
	             mmd_kernel_type='rbf', mmd_kernel_mul=2., mmd_num_kernels=20, dropout=0.2):
		super(Inception, self).__init__()
		self.genotypes = genotypes
		self.criterion = criterion
		self.classes = classes
		self.num_blocks = num_blocks
		self.C = C
		self.num_task = num_task
		self.mmd_kernel_type = mmd_kernel_type
		self.mmd_kernel_mul = mmd_kernel_mul
		self.mmd_num_kernels = mmd_num_kernels
		self.dropout_rate = dropout
		self.MMD_criterion = MMD_loss(self.mmd_kernel_type, self.mmd_kernel_mul, self.mmd_num_kernels)
		
		self.stem_conv = nn.Sequential(
			nn.Conv1d(1, self.C, kernel_size=5, padding=2, stride=1, bias=False),
			nn.BatchNorm1d(self.C),
			nn.ReLU(),
		)
		
		self.block = nn.ModuleList([])
		for i in range(num_blocks):
			if i % 2 != 0:
				reduction = True
				self.block.append(Block(self.C, self.C * 2, reduction, genotypes))
				self.C *= 2
			else:
				reduction = False
				self.block.append(Block(self.C, self.C, reduction, genotypes))
		
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
		self.dropout = nn.Dropout(self.dropout_rate)
		self.classify = nn.Linear(self.C, classes)
		
		self._initialize_weights()
	
	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm1d):
				if m.weight is not None:
					nn.init.ones_(m.weight)
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	
	def forward(self, x, k=-1):
		x = self.stem_conv(x)
		for i in range(self.num_blocks):
			x = self.block[i](x)
		x = self.global_avg_pool(x)
		features = x.view(x.shape[0], -1)
		features = features * torch.Tensor(self.genotypes.features)#.cuda()
		x = self.dropout(features)
		out = self.classify(x)
		return out, features
	
	def clf_loss(self, logits, target):
		return self.criterion(logits, target)
	
	def mmd_loss(self, fc_src, fc_tar):
		return self.MMD_criterion(fc_src, fc_tar)
	


if __name__ == '__main__':
	criterion = nn.CrossEntropyLoss()
	model = Inception(cwru_src_0, criterion)
	model_size = count_parameters_in_MB(model)
	print(model)
	print('model size/M: ', model_size)
	inputs = torch.rand((8, 1, 1000))
	out, features = model(inputs)
	loss = model.mmd_loss(features, features, task_id=0)
	print('output: ', out)
	print('mmd loss: ', loss)
