import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Architect(object):
	def __init__(self, model, args):
		self.network_momentum = args.momentum
		self.network_weight_decay = args.weight_decay
		self.model = model
		self.args = args
		self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
		                                  lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
	
	def step(self, src_val_x, src_val_y, tar_x_list):
		self.optimizer.zero_grad()
		
		src_pred, src_fc = self.model(src_val_x)
		clf_loss = self.model.clf_loss(src_pred, src_val_y)
		aux_loss = self.model.aux_loss()
		mmd_loss = 0.
		for task_id, tar_x in enumerate(tar_x_list):
			_, tar_fc = self.model(tar_x)
			mmd_loss += self.model.mmd_loss(src_fc, tar_fc)
		mmd_loss /= self.args.num_tasks
		loss = clf_loss + self.args.labda * mmd_loss + self.args.aux_labda * aux_loss
		loss.backward()
		self.optimizer.step()

		


