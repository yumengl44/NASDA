import os
import sys
import time
import glob
import math
import numpy as np
import torch
import utils
import copy
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from Inception_search import Inception
from architect import Architect
from visualize import plot_architecture, plot_alphas, plot_acc_loss
torch.set_printoptions(profile="full")


def train(epoch, src_train_loader, src_val_loader, src_test_loader, tar_train_loaders, tar_test_loaders,
          device, model, architect, optimizer):
	clf_loss_meter = utils.AvgrageMeter()
	mmd_loss_meter = utils.AvgrageMeter()
	acc_meter = utils.AvgrageMeter()

	src_train_iter = iter(src_train_loader)
	src_val_iter = iter(src_val_loader)
	tar_train_iters = []
	for i in range(args.num_tasks):
		tar_train_iters.append(iter(tar_train_loaders[i]))
	n_batch = min(len(src_train_loader), min([len(loader) for loader in tar_train_loaders]))

	for step in range(n_batch):
		model.train()
		src_x, src_y = next(src_train_iter)
		src_x, src_y = src_x.to(device).float(), src_y.to(device).long()
		src_val_x, src_val_y = next(src_val_iter)
		src_val_x, src_val_y = src_val_x.to(device).float(), src_val_y.to(device).long()
		
		tar_x_list = []
		for tar_iter in tar_train_iters:
			tar_x, _ = next(tar_iter)
			tar_x = tar_x.to(device).float()
			tar_x_list.append(tar_x)
		n = src_x.size(0)
		
		# update alphas of network
		architect.step(src_val_x, src_val_y, tar_x_list)
		
		# update weights of network
		optimizer.zero_grad()
		src_pred, src_fc = model(src_x, -1)
		clf_loss = model.clf_loss(src_pred, src_y)
		aux_loss = model.aux_loss()
		mmd_loss = 0.
		for task_id, tar_x in enumerate(tar_x_list):
			_, tar_fc = model(tar_x, task_id)
			mmd_loss += model.mmd_loss(src_fc, tar_fc)
		mmd_loss /= args.num_tasks
		loss = clf_loss + args.labda * mmd_loss + args.aux_labda * aux_loss
		loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
		optimizer.step()
		
		# update meter
		prec1 = utils.accuracy(src_pred, src_y)[0]
		acc_meter.update(prec1.item(), n)
		clf_loss_meter.update(clf_loss.item(), n)
		mmd_loss_meter.update(mmd_loss.item(), n)
		
		print('\rEpoch[{}/{}], step {}/{}, clf loss: {:.6}, mmd loss: {:.6}, aux_loss: {:.6}, train acc: {:.6}'.format(
			epoch + 1, args.epochs, step + 1, n_batch, clf_loss_meter.avg, mmd_loss_meter.avg, aux_loss.item(), acc_meter.avg
		), end='')
		
	return acc_meter.avg, clf_loss_meter.avg, mmd_loss_meter.avg


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
	criterion = nn.CrossEntropyLoss().cuda()
	model = Inception(criterion, classes=args.classes, num_blocks=args.num_blocks, C=args.C,
	                  num_task=args.num_tasks, mmd_kernel_type=args.mmd_kernel_type,
	                  mmd_kernel_mul=args.mmd_kernel_mul, mmd_num_kernels=args.mmd_num_kernels)
	model = model.cuda()
	print('param size = {} MB'.format(utils.count_parameters_in_MB(model)))
	
	architect = Architect(model, args)
	optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	# optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
	
	# data
	# src
	src_data_path = '{}{}/work_{}.csv'.format(args.data_dir, args.dataset, args.src)
	data_loaders, _ = utils.load_datasets(args, src_data_path, val_size=0.5)
	src_train_loader = data_loaders['train_dataset']
	src_val_loader = data_loaders['val_dataset']
	src_test_loader = data_loaders['test_dataset']
	# tar
	tar_train_loaders, tar_test_loaders = [], []
	for tar in range(args.num_tasks + 1):
		if tar == args.src:
			continue
		tar_data_path = '{}{}/work_{}.csv'.format(args.data_dir, args.dataset, tar)
		data_loaders, _ = utils.load_datasets(args, tar_data_path, val_size=0.01)
		tar_train_loader = data_loaders['train_dataset']
		tar_test_loader = data_loaders['test_dataset']
		tar_train_loaders.append(tar_train_loader)
		tar_test_loaders.append(tar_test_loader)
	
	normal_list, reduce_list, alphas_features_list = [], [], []
	train_acc_list, mmd_loss_list, aux_loss_list = [], [], []
	init_aux_labda = copy.deepcopy(args.aux_labda)
	for epoch in range(args.epochs):
		t1 = time.time()
		lr = scheduler.get_lr()[0]
		print('lr {}'.format(lr))
		
		genotype = model.genotype(args.mmd_threshold)
		print(genotype)
		
		print('alphas_normal\n', F.softmax(model.alphas_normal, dim=-1))
		print('alphas_reduce\n', F.softmax(model.alphas_reduce, dim=-1))
		print('alphas_features\n', F.sigmoid(model.alphas_features))
		normal_list.append(F.softmax(model.alphas_normal, dim=-1).detach().cpu().numpy())
		reduce_list.append(F.softmax(model.alphas_reduce, dim=-1).detach().cpu().numpy())
		alphas_features_list.append(F.sigmoid(model.alphas_features).detach().cpu().numpy())
		
		# training
		args.aux_labda = init_aux_labda * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
		print('aux_labda: {}'.format(args.aux_labda))
		train_acc, train_loss, mmd_loss = train(epoch, src_train_loader, src_val_loader, src_test_loader,
		                                        tar_train_loaders, tar_test_loaders, device,
		                                        model, architect, optimizer)
		train_acc_list.append(train_acc)
		mmd_loss_list.append(mmd_loss)
		scheduler.step(epoch)
		
		# validation
		valid_acc, valid_loss = infer(src_val_loader, model, criterion, -1)
		print(', val acc: {:.6}, val loss: {:.6}'.format(valid_acc, valid_loss))
		
		# save
		utils.save(model, os.path.join(args.save, 'weights.pt'))
		
		t2 = time.time()
		print('this epoch time/min: {:.4}'.format((t2 - t1) / 60))
	
	# plot
	plot_alphas(args.save, normal_list, reduce_list, alphas_features_list)
	plot_acc_loss(args.save, train_acc_list, mmd_loss_list)
	# save
	alphas_features_list = np.vstack(alphas_features_list)
	np.save(os.path.join(args.save, 'alphas_features.npy'), alphas_features_list)
	print(model.genotype(args.mmd_threshold))

if __name__ == '__main__':
	parser = argparse.ArgumentParser("PHM")
	# data and task
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
	parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
	# model
	parser.add_argument('--C', type=int, default=18, help='num of init channels')
	parser.add_argument('--num_blocks', type=int, default=8, help='total number of blocks')
	parser.add_argument('--classes', type=int, default=4, help='classes number')
	parser.add_argument('--mmd_kernel_type', type=str, default='rbf', help='kernel type of MMD')
	parser.add_argument('--mmd_kernel_mul', type=float, default=2.0, help='kernel mul of MMD')
	parser.add_argument('--mmd_num_kernels', type=int, default=20, help='number of kernels for MMD')
	parser.add_argument('--mmd_threshold', type=float, default=0.5, help='threshold of mmd alphas')
	parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
	# train
	parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
	parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
	parser.add_argument('--learning_rate_min', type=float, default=0., help='min learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
	parser.add_argument('--arch_learning_rate', type=float, default=1e-2, help='learning rate for arch encoding')
	parser.add_argument('--arch_weight_decay', type=float, default=0., help='weight decay for arch encoding')
	parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
	parser.add_argument('--labda', type=float, default=0.1, help='mmd loss lambda')
	parser.add_argument('--aux_labda', type=float, default=0.1, help='mmd loss lambda')
	
	
	args = parser.parse_args()
	args.save = './logs/' + args.save
	
	if args.dataset == 'CWRU':
		args.classes = 4
	if args.dataset == 'Paderborn':
		args.classes = 3
	
	print(args)
	
	# nohup python -u train_search.py --dataset CWRU --src 0 --seed 10 --save cwru_src_0_seed_10 --gpu 0 > cwru_src_0_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset CWRU --src 1 --seed 10 --save cwru_src_1_seed_10 --gpu 1 > cwru_src_1_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset CWRU --src 2 --seed 10 --save cwru_src_2_seed_10 --gpu 2 > cwru_src_2_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset CWRU --src 3 --seed 10 --save cwru_src_3_seed_10 --gpu 3 > cwru_src_3_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset Paderborn --src 0 --seed 10 --save paderborn_src_0_seed_10 --gpu 4 > paderborn_src_0_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset Paderborn --src 1 --seed 10 --save paderborn_src_1_seed_10 --gpu 5 > paderborn_src_1_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset Paderborn --src 2 --seed 10 --save paderborn_src_2_seed_10 --gpu 6 > paderborn_src_2_seed_10.log 2>&1 &
	# nohup python -u train_search.py --dataset Paderborn --src 3 --seed 10 --save paderborn_src_3_seed_10 --gpu 7 > paderborn_src_3_seed_10.log 2>&1 &
	
	utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
	main()

