import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import Inception


def train(epoch, device, model, src_train_loader, tar_train_loaders, optimizer):
	clf_loss_meter = utils.AvgrageMeter()
	mmd_loss_meter = utils.AvgrageMeter()
	train_acc_meter = utils.AvgrageMeter()
	
	src_train_iter = iter(src_train_loader)
	tar_train_iters = []
	for i in range(args.num_tasks):
		tar_train_iters.append(iter(tar_train_loaders[i]))
	n_batch = min(len(src_train_loader), min([len(loader) for loader in tar_train_loaders]))
	
	for step in range(n_batch):
		model.train()
		src_x, src_y = next(src_train_iter)
		src_x, src_y = src_x.to(device).float(), src_y.to(device).long()
		
		tar_x_list = []
		for tar_iter in tar_train_iters:
			tar_x, _ = next(tar_iter)
			tar_x = tar_x.to(device).float()
			tar_x_list.append(tar_x)
		n = src_x.size(0)
		
		optimizer.zero_grad()
		src_pred, src_fc = model(src_x, -1)
		clf_loss = model.clf_loss(src_pred, src_y)
		mmd_loss = 0.
		for task_id, tar_x in enumerate(tar_x_list):
			_, tar_fc = model(tar_x, task_id)
			# fc_weights = torch.Tensor(model.genotypes.features[task_id]).cuda()
			# tar_fc = tar_fc * fc_weights   # features selection
			mmd_loss += model.mmd_loss(src_fc, tar_fc)
		loss = clf_loss + args.labda * mmd_loss
		loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
		optimizer.step()
		
		# update meter
		prec1 = utils.accuracy(src_pred, src_y)[0]
		train_acc_meter.update(prec1.item(), n)
		clf_loss_meter.update(clf_loss.item(), n)
		mmd_loss_meter.update(mmd_loss.item(), n)
		
		print('\rEpoch[{}/{}], step {}/{}, clf loss: {:.6}, mmd loss: {:.6}, train acc: {:.6}'.format(
			epoch + 1, args.epochs, step + 1, n_batch, clf_loss_meter.avg, mmd_loss_meter.avg, train_acc_meter.avg
		), end='')
	
	return train_acc_meter.avg, clf_loss_meter.avg, mmd_loss_meter.avg


def infer(data_loader, model, criterion, task_id):
	objs = utils.AvgrageMeter()
	top1 = utils.AvgrageMeter()
	model.eval()
	
	for step, (input, target) in enumerate(data_loader):
		input, target = input.cuda().float(), target.cuda().long()
		
		logits, _ = model(input, task_id)
		loss = criterion(logits, target)
		
		prec1 = utils.accuracy(logits, target)[0]
		n = input.size(0)
		objs.update(loss.item(), n)
		top1.update(prec1.item(), n)
	
	return top1.avg, objs.avg


def main():
	if not torch.cuda.is_available():
		print('no gpu device available')
		sys.exit(1)
	
	np.random.seed(args.seed)
	torch.cuda.set_device(args.gpu)
	cudnn.benchmark = True
	cudnn.enabled = True
	device = torch.device("cuda")
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	print('gpu device = {}'.format(args.gpu))
	
	# model
	genotype = eval("genotypes.%s" % args.arch)
	criterion = nn.CrossEntropyLoss().cuda()
	model = Inception(genotype, criterion, classes=args.classes, num_blocks=args.num_blocks, C=args.C,
	                  num_task=args.num_tasks, mmd_kernel_type=args.mmd_kernel_type,
	                  mmd_kernel_mul=args.mmd_kernel_mul, mmd_num_kernels=args.mmd_num_kernels, dropout=args.dropout)
	model = model.cuda()
	print('param size = {} MB'.format(utils.count_parameters_in_MB(model)))
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
	                                                       eta_min=args.learning_rate_min)
	
	# data
	# src
	src_data_path = '{}{}/work_{}.csv'.format(args.data_dir, args.dataset, args.src)
	data_loaders, _ = utils.load_datasets(args, src_data_path, val_size=0.1)
	src_train_loader = data_loaders['train_dataset']
	src_val_loader = data_loaders['val_dataset']
	src_test_loader = data_loaders['test_dataset']
	# tar
	tar_train_loaders, tar_test_loaders = [], []
	for tar in range(args.num_tasks + 1):
		if tar == args.src:
			continue
		tar_data_path = '{}{}/work_{}.csv'.format(args.data_dir, args.dataset, tar)
		data_loaders, _ = utils.load_datasets(args, tar_data_path, val_size=0.1)
		tar_train_loader = data_loaders['train_dataset']
		tar_test_loader = data_loaders['test_dataset']
		tar_train_loaders.append(tar_train_loader)
		tar_test_loaders.append(tar_test_loader)
	
	# train
	for epoch in range(args.epochs):
		t1 = time.time()
		train(epoch, device, model, src_train_loader, tar_train_loaders, optimizer)
		
		src_val_acc, _ = infer(src_val_loader, model, criterion, -1)
		print(', src val acc: {:.6}'.format(src_val_acc))
		tar_test_acc = []
		for i in range(args.num_tasks):
			tar_test_acc.append(infer(tar_test_loaders[i], model, criterion, i)[0])
		print('tar test acc: ', tar_test_acc)
		
		utils.save(model, os.path.join(args.save, 'model.pt'))
		scheduler.step(epoch)
		t2 = time.time()
		print('this epoch time/min: {:.4}'.format((t2 - t1) / 60))
	tar_test_acc = []
	for i in range(args.num_tasks):
		tar_test_acc.append(infer(tar_test_loaders[i], model, criterion, i)[0])
	print('this fold test tar acc: ', tar_test_acc)
	return tar_test_acc




if __name__ == '__main__':
	parser = argparse.ArgumentParser("train")
	# data and task
	parser.add_argument('--arch', type=str, default='cwru_src_0', help='architecture name')
	parser.add_argument('--save', type=str, default='search', help='save path')
	parser.add_argument('--dataset', type=str, default='CWRU', help='target dataset, [CWRU, Paderborn]')
	parser.add_argument('--data_dir', default='/work/lixudong/phd_codework/data/', type=str, help='the path of dataset')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--num_tasks', type=int, default=3, help='number of target tasks')
	parser.add_argument('--src', type=int, default=0, help='source domain')
	parser.add_argument('--k', type=int, default=0, help='k-fold')
	parser.add_argument('--seed', type=int, default=10, help='random seed')
	parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
	parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
	# model
	parser.add_argument('--C', type=int, default=18, help='num of init channels')
	parser.add_argument('--num_blocks', type=int, default=8, help='total number of blocks')
	parser.add_argument('--classes', type=int, default=4, help='classes number')
	parser.add_argument('--mmd_kernel_type', type=str, default='rbf', help='kernel type of MMD')
	parser.add_argument('--mmd_kernel_mul', type=float, default=2.0, help='kernel mul of MMD')
	parser.add_argument('--mmd_num_kernels', type=int, default=20, help='number of kernels for MMD')
	parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
	# train
	parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
	parser.add_argument('--learning_rate_min', type=float, default=0., help='min learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
	parser.add_argument('--dropout', type=float, default=0., help='dropout')
	parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
	parser.add_argument('--labda', type=float, default=0.1, help='mmd loss lambda')

	args = parser.parse_args()
	args.save = './logs/' + args.arch
	if args.dataset == 'CWRU':
		args.classes = 4
	if args.dataset == 'Paderborn':
		args.classes = 3
	print(args)
	print(eval("genotypes.%s" % args.arch))
	
	if not os.path.exists(args.save):
		utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
	
	# nohup python -u train.py --dataset CWRU --src 0 --seed 10 --arch cwru_src_0_seed_10 --gpu 0 > train_cwru_src_0_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset CWRU --src 1 --seed 10 --arch cwru_src_1_seed_10 --gpu 1 > train_cwru_src_1_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset CWRU --src 2 --seed 10 --arch cwru_src_2_seed_10 --gpu 2 > train_cwru_src_2_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset CWRU --src 3 --seed 10 --arch cwru_src_3_seed_10 --gpu 3 > train_cwru_src_3_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset Paderborn --src 0 --seed 10 --arch paderborn_src_0_seed_10 --gpu 4 > train_paderborn_src_0_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset Paderborn --src 1 --seed 10 --arch paderborn_src_1_seed_10 --gpu 5 > train_paderborn_src_1_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset Paderborn --src 2 --seed 10 --arch paderborn_src_2_seed_10 --gpu 6 > train_paderborn_src_2_seed_10.log 2>&1 &
	# nohup python -u train.py --dataset Paderborn --src 3 --seed 10 --arch paderborn_src_3_seed_10 --gpu 7 > train_paderborn_src_3_seed_10.log 2>&1 &
	test_acc_list = []
	for k in range(5):
		print('{} model, [{}-th fold]'.format(args.arch, k) + '-' * 20)
		args.k = k
		test_acc = main()
		test_acc_list.append(test_acc)
	print('-' * 50)
	test_acc_list = np.array(test_acc_list).reshape((-1, args.num_tasks)).T
	for i in range(args.num_tasks):
		print(','.join([str(a) for a in test_acc_list[i]]))
	print('Target task avg test acc: ')
	for i in range(args.num_tasks):
		print(np.mean(test_acc_list[:, i]))

