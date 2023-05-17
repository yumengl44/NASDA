import os
import sys
import numpy as np
import genotypes
from graphviz import Digraph
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from genotypes import *



def plot_architecture(genotype, filename):
	g = Digraph(
		format='pdf',
		edge_attr=dict(fontsize='20', fontname="times"),
		node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
		engine='dot')
	g.body.extend(['rankdir=LR'])
	
	g.node("input", fillcolor='darkseagreen2')
	g.node("output", fillcolor='darkseagreen2')
	
	for i in range(len(genotype)):
		op = genotype[i]
		g.edge('input', 'output', label=op, fillcolor="gray")
	
	g.render(filename, view=False)


def plot_alphas(save, normal_list, reduce_list, alphas_features_list):
	# normal list
	plt.figure(figsize=(9, 6))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	
	for i in range(4):
		data = []
		for normal in normal_list:
			data.append(normal[i])
		data = np.array(data).reshape((len(normal_list), -1))
		
		plt.subplot(2, 2, i + 1)
		for j in range(data.shape[1]):
			plt.plot(data[:, j], linewidth=1, label=INCEPTION[j])
		plt.xlabel('Epoch')
		plt.ylabel('softmax($alphas$)')
		plt.title('Branch {}'.format(i + 1))
	plt.legend(loc=2, bbox_to_anchor=(1.05,1.0), borderaxespad=0.)
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save, 'alphas_normal.png'), dpi=600)
	plt.savefig(os.path.join(save, 'alphas_normal.eps'))
	
	# reduce list
	plt.figure(figsize=(9, 3))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	
	for i in range(2):
		data = []
		for reduce in reduce_list:
			data.append(reduce[i])
		data = np.array(data).reshape((len(reduce_list), -1))
		
		plt.subplot(1, 2, i + 1)
		for j in range(data.shape[1]):
			plt.plot(data[:, j], linewidth=1, label=INCEPTION[j])
		plt.xlabel('Epoch')
		plt.ylabel('softmax($alphas$)')
		plt.title('Branch {}'.format(i + 1))
	plt.legend(loc=2, bbox_to_anchor=(1.05, 1.), borderaxespad=0.)
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save, 'alphas_reduce.png'), dpi=600)
	plt.savefig(os.path.join(save, 'alphas_reduce.eps'))

	# features weights list
	plt.figure(figsize=(12, 3))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	for i in range(1):
		data = []
		for alphas in alphas_features_list:
			data.append(alphas[i])
		data = np.array(data).reshape((len(alphas_features_list), -1))

		plt.subplot(1, 3, i + 1)
		for j in range(data.shape[1]):
			plt.plot(data[:, j], linewidth=1)
		plt.hlines(0.5, 0, data.shape[0] - 1, colors='grey', linestyles='--', linewidth=1.5)
		plt.xlabel('Epoch')
		plt.ylabel('sigmoid($alphas$)')
		plt.title('Target task {}'.format(i + 1))
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save, 'alphas_features.png'), dpi=600)
	plt.savefig(os.path.join(save, 'alphas_features.eps'))
	
	#
	plt.figure(figsize=(13, 6))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	data = np.vstack(alphas_features_list)
	plt.pcolor(data, cmap=plt.cm.Oranges, edgecolors='k', alpha=0.6)
	plt.colorbar()
	plt.clim(0., 1.)
	plt.xlabel('Features')
	plt.ylabel('Epoch')
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save, 'alphas_features_2.png'), dpi=600)
	
	# plot heatmap
	plt.figure(figsize=(12, 1))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	data = alphas_features_list[-1]
	data[data >= 0.5] = 1
	data[data < 0.5] = 0
	plt.pcolor(data, cmap=plt.cm.Oranges, edgecolors='k', alpha=0.6)
	plt.yticks([])
	plt.xlabel('Features')
	# plt.ylabel('Task')
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save, 'final_features.png'), dpi=600)
	plt.savefig(os.path.join(save, 'final_features.eps'))
		


def plot_acc_loss(save, train_acc_list, mmd_loss_list):
	plt.figure(figsize=(7, 3))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	
	plt.subplot(1, 2, 1)
	plt.plot(train_acc_list, linewidth=1)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy/%')
	plt.title('Train accuracy')
	
	plt.subplot(1, 2, 2)
	plt.plot(mmd_loss_list, linewidth=1)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('MMD loss')
	
	plt.tight_layout()
	plt.show()
	plt.savefig(os.path.join(save, 'acc_loss.png'), dpi=600)
	plt.savefig(os.path.join(save, 'acc_loss.eps'))
	


if __name__ == '__main__':
	# if len(sys.argv) != 2:
	# 	print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
	# 	sys.exit(1)
	#
	# genotype_name = sys.argv[1]
	# try:
	# 	genotype = eval('genotypes.{}'.format(genotype_name))
	# except AttributeError:
	# 	print("{} is not specified in genotypes.py".format(genotype_name))
	# 	sys.exit(1)
	
	# for genotype in ['cwru_src_0_seed_300', 'cwru_src_1_seed_400', 'cwru_src_2_seed_800', 'cwru_src_3_seed_800']:
	# 	plot_architecture(eval(genotype).normal, "./figures/{}_normal".format(genotype))
	# 	plot_architecture(eval(genotype).reduce, "./figures/{}_reduction".format(genotype))
	# for genotype in ['paderborn_src_0_seed_800', 'paderborn_src_1_seed_400', 'paderborn_src_2_seed_500', 'paderborn_src_3_seed_800']:
	# 	plot_architecture(eval(genotype).normal, "./figures/{}_normal".format(genotype))
	# 	plot_architecture(eval(genotype).reduce, "./figures/{}_reduction".format(genotype))
	for genotype in ['real_src_0_seed_70', 'real_src_1_seed_90', 'real_src_2_seed_80', 'real_src_3_seed_20']:
		plot_architecture(eval(genotype).normal, "./figures/{}_normal".format(genotype))
		plot_architecture(eval(genotype).reduce, "./figures/{}_reduction".format(genotype))

