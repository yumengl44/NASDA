import torch
import torch.nn as nn
import torch.nn.functional as F
from Inception_operations import *
from torch.autograd import Variable
from genotypes import INCEPTION
from genotypes import Genotype
from utils import MMD_loss, count_parameters_in_MB


class Branch(nn.Module):
	def __init__(self, C, stride):
		super(Branch, self).__init__()
		self._ops = nn.ModuleList()
		for primitive in INCEPTION:
			op = OPS[primitive](C, stride, False)
			if 'pool' in primitive:
				op = nn.Sequential(op, nn.BatchNorm1d(C, affine=False))
			self._ops.append(op)
	
	def forward(self, x, weights):
		return sum(w * op(x) for w, op in zip(weights, self._ops))



class Block(nn.Module):
	def __init__(self, C_in, C_out, reduction):
		super(Block, self).__init__()
		self.reduction = reduction
		
		self.conv_list = nn.ModuleList()
		self.branch_list = nn.ModuleList()
		self.num_branch = 2 if reduction else 4
		for i in range(self.num_branch):
			self.conv_list.append(
				nn.Sequential(
					nn.Conv1d(C_in, C_out // self.num_branch, kernel_size=1, stride=1, bias=False),
					nn.BatchNorm1d(C_out // self.num_branch, affine=True),
					nn.ReLU()
				)
			)
			stride = 2 if reduction else 1
			self.branch_list.append(Branch(C_out // self.num_branch, stride))
		
		self.conv = nn.Sequential(
			nn.Conv1d(C_out // self.num_branch * self.num_branch, C_out, kernel_size=1, padding=0, stride=1, bias=False),
			nn.BatchNorm1d(C_out),
			nn.ReLU()
		)
			
	
	def forward(self, x, weights):
		temp = []
		for i in range(self.num_branch):
			s = self.conv_list[i](x)
			s = self.branch_list[i](s, weights[i])
			temp.append(s)
		s = torch.cat(temp, dim=1)
		out = self.conv(s)
		
		return out


class Inception(nn.Module):
	def __init__(self, criterion, classes=4, num_blocks=8, C=16, num_task=3,
	             mmd_kernel_type='rbf', mmd_kernel_mul=2., mmd_num_kernels=20):
		super(Inception, self).__init__()
		self.criterion = criterion
		self.classes = classes
		self.num_blocks = num_blocks
		self.C = C
		self.num_task = num_task
		self.mmd_kernel_type = mmd_kernel_type
		self.mmd_kernel_mul = mmd_kernel_mul
		self.mmd_num_kernels = mmd_num_kernels
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
				self.block.append(Block(self.C, self.C * 2, reduction))
				self.C *= 2
			else:
				reduction = False
				self.block.append(Block(self.C, self.C, reduction))
		
		self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
		self.classify = nn.Linear(self.C, classes)
	
		self._initialize_weights()
		self._initialize_alphas()
	
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
	
	def _initialize_alphas(self):
		num_ops = len(INCEPTION)
		
		self.alphas_normal = Variable(1e-3 * torch.randn(4, num_ops).cuda(), requires_grad=True)
		self.alphas_reduce = Variable(1e-3 * torch.randn(2, num_ops).cuda(), requires_grad=True)
		self.alphas_features = Variable(1e-3 * torch.randn(1, self.C).cuda(), requires_grad=True)
		self._arch_parameters = [
			self.alphas_normal,
			self.alphas_reduce,
			self.alphas_features
		]
	
	def forward(self, x, k=-1):
		x = self.stem_conv(x)
		for i in range(self.num_blocks):
			if self.block[i].reduction:
				weights = F.softmax(self.alphas_reduce, dim=-1)
			else:
				weights = F.softmax(self.alphas_normal, dim=-1)
			x = self.block[i](x, weights)
		x = self.global_avg_pool(x)
		features = x.view(x.shape[0], -1)
		features = features * F.sigmoid(self.alphas_features)
		out = self.classify(features)
		
		return out, features
	
	def clf_loss(self, logits, target):
		return self.criterion(logits, target)
	
	def mmd_loss(self, fc_src, fc_tar):
		return self.MMD_criterion(fc_src, fc_tar)
	
	def aux_loss(self):
		loss = -F.mse_loss(F.sigmoid(self.alphas_features), torch.tensor(0.5, requires_grad=False).cuda())
		# loss = -torch.mean(torch.abs(F.sigmoid(self.alphas_features) - 0.5))
		# loss = torch.sum(torch.abs(F.sigmoid(self.alphas_features))) / self.num_task
		return loss
	
	def new(self):
		model_new = Inception(self.classes, self.num_blocks, self.C, self.dropout_rate).cuda()
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data.copy_(y.data)
		return model_new
	
	def arch_parameters(self):
		return self._arch_parameters
	
	def genotype(self, mmd_threshold):
		def _parse(weights):
			gene = []
			for i in range(weights.shape[0]):
				W = weights[i].copy()
				k_best = None
				for k in range(len(W)):
					if k_best is None or W[k] > W[k_best]:
						k_best = k
				gene.append(INCEPTION[k_best])
			return gene
		
		gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
		gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
		
		gene_features = F.sigmoid(self.alphas_features).data.cpu().numpy()
		gene_features[gene_features >= mmd_threshold] = 1
		gene_features[gene_features < mmd_threshold] = 0
		
		genotype = Genotype(
			normal=gene_normal,
			reduce=gene_reduce,
			features=gene_features.tolist()
		)
		return genotype



if __name__ == '__main__':
	criterion = nn.CrossEntropyLoss()
	model = Inception(criterion)
	model_size = count_parameters_in_MB(model)
	print(model)
	print('model size/M: ', model_size)
	inputs = torch.rand((8, 1, 1000))
	out, features = model(inputs)
	loss = model.mmd_loss(features, features, task_id=0)
	print('output: ', out)
	print('mmd loss: ', loss)
	print(model.genotype())




