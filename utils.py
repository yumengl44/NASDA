import os
import numpy as np
import torch
import torch.nn as nn
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split



class AvgrageMeter(object):
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.avg = 0.
		self.sum = 0.
		self.cnt = 0.
	
	def update(self, val, n=1):
		self.sum += val * n
		self.cnt += n
		self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0/batch_size))
	return res


class MMD_loss(nn.Module):
	def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=20):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		self.kernel_type = kernel_type

	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0]) + int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		total0 = total.unsqueeze(0).expand(
		    int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(
		    int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0 - total1) ** 2).sum(2)
		if fix_sigma is not None:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def linear_mmd2(self, f_of_X, f_of_Y):
		loss = 0.0
		delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
		loss = delta.dot(delta.T)
		return loss

	def forward(self, source, target):
		if self.kernel_type == 'linear':
			return self.linear_mmd2(source, target)
		elif self.kernel_type == 'rbf':
			batch_size = int(source.size()[0])
			kernels = self.guassian_kernel(
			    source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
			# with torch.no_grad():
			XX = torch.mean(kernels[:batch_size, :batch_size])
			YY = torch.mean(kernels[batch_size:, batch_size:])
			XY = torch.mean(kernels[:batch_size, batch_size:])
			YX = torch.mean(kernels[batch_size:, :batch_size])
			loss = torch.mean(XX + YY - XY - YX)
			# torch.cuda.empty_cache()
			return loss


class MyDataset(Dataset):
	def __init__(self, data, index, show=True):
		self.x, self.y = data[index, np.newaxis, :-1], data[index, -1]
		self.len = self.x.shape[0]
		if show:
			print('dataset shape: {}, {}'.format(self.x.shape, self.y.shape))
		del data
	
	def __getitem__(self, index):
		return self.x[index], self.y[index]
	
	def __len__(self):
		return self.len
	
	def get_data(self):
		return self.x, self.y
	
	def get_random_data(self, seed=26):
		np.random.seed(seed)
		index = np.random.permutation(range(len(self.x)))
		return self.x[index], self.y[index]


def load_datasets(args, data_path, val_size, show=True):
	data = np.loadtxt(data_path)
	
	# Cross-validation
	folds = []
	kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
	for (train_index, test_index) in kf.split(data, data[:, -1]):
		train_index, val_index = train_test_split(train_index, test_size=val_size, random_state=args.seed)
		folds.append({'train': train_index, 'val': val_index, 'test': test_index})
	
	# dataset
	train_dataset = MyDataset(data, index=folds[args.k]['train'], show=show)
	val_dataset = MyDataset(data, index=folds[args.k]['val'], show=show)
	test_dataset = MyDataset(data, index=folds[args.k]['test'], show=show)
	datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
	
	# dataloader
	data_loaders = {}
	data_loaders['train_dataset'] = Data.DataLoader(dataset=train_dataset, pin_memory=True, shuffle=True,
	                                                batch_size=args.batch_size, num_workers=args.num_workers,
	                                                drop_last=True)
	data_loaders['val_dataset'] = Data.DataLoader(dataset=val_dataset, pin_memory=True, shuffle=False,
	                                              batch_size=args.batch_size, num_workers=args.num_workers,
	                                              drop_last=True)
	data_loaders['test_dataset'] = Data.DataLoader(dataset=test_dataset, pin_memory=True, shuffle=False,
	                                               batch_size=args.batch_size, num_workers=args.num_workers,
	                                               drop_last=False)
	
	return data_loaders, datasets



def count_parameters_in_MB(model):
	return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
	filename = os.path.join(save, 'checkpoint.pth.tar')
	torch.save(state, filename)
	if is_best:
		best_filename = os.path.join(save, 'model_best.pth.tar')
		shutil.copyfile(filename, best_filename)


def save(model, model_path):
	torch.save(model.state_dict(), model_path)


def load(model, model_path):
	model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)
	print('Experiment dir : {}'.format(path))
	
	if scripts_to_save is not None:
		srcipts_path = os.path.join(path, 'scripts')
		if not os.path.exists(srcipts_path):
			os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)

