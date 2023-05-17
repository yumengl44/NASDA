import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from sklearn.manifold import TSNE
import copy
import warnings
warnings.filterwarnings("ignore")
import utils
from model import Inception
from model_for_tsne import InceptionModel, InceptionTransfer
from genotypes import *


class args:
	seed = 16
	k = 0
	batch_size = 50
	num_workers = 1



def plot_tsne(dataset):
	classes = 4 if dataset == 'CWRU' else 3
	# data_path = '/work/lixudong/phd_codework/data/'
	data_path = 'F:/AI/code_work/phd_codework/data/'
	# data
	src_data_path = data_path + '{}/work_0.csv'.format(dataset)
	_, src_dataset = utils.load_datasets(args, src_data_path, val_size=0.1)
	src_x, src_y = src_dataset['test'].get_random_data()
	src_x, src_y = src_x[:args.batch_size], src_y[:args.batch_size].astype(np.int)
	
	tar_data_path = data_path + '{}/work_1.csv'.format(dataset)
	_, tar_dataset = utils.load_datasets(args, tar_data_path, val_size=0.1)
	tar_x, tar_y = tar_dataset['test'].get_random_data()
	tar_x_1, tar_y_1 = tar_x[:args.batch_size], tar_y[:args.batch_size].astype(np.int)
	
	tar_data_path = data_path + '{}/work_2.csv'.format(dataset)
	_, tar_dataset = utils.load_datasets(args, tar_data_path, val_size=0.1)
	tar_x, tar_y = tar_dataset['test'].get_random_data()
	tar_x_2, tar_y_2 = tar_x[:args.batch_size], tar_y[:args.batch_size].astype(np.int)
	
	tar_data_path = data_path + '{}/work_3.csv'.format(dataset)
	_, tar_dataset = utils.load_datasets(args, tar_data_path, val_size=0.1)
	tar_x, tar_y = tar_dataset['test'].get_random_data()
	tar_x_3, tar_y_3 = tar_x[:args.batch_size], tar_y[:args.batch_size].astype(np.int)
	
	# model
	# source_only_path = '/work/lixudong/phd_codework/chapter4/domain_adaptation/models/source_only_inception_{}_0_1.pt.tar'.format(dataset)
	# baseline_path = '/work/lixudong/phd_codework/chapter4/domain_adaptation/models/normal/mmd/inception_{}_0_1.pt.tar'.format(dataset)
	# ffcnn_path = '/work/lixudong/phd_codework/chapter4/domain_adaptation/models/ffcnn/mmd/inception_{}_0_1.pt.tar'.format(dataset)
	# dann_path = '/work/lixudong/phd_codework/chapter4/domain_adaptation/models/normal/dann/inception_{}_0_1.pt.tar'.format(dataset)
	# darts_path = '/work/lixudong/PHM_NAS_domain_adaptation_new/logs/{}_src_0_seed_{}0/model.pt'.format(dataset.lower(), 30 if dataset == 'CWRU' else 80)
	
	source_only_path = 'F:/AI/code_work/phd_codework/chapter4/domain_adaptation/models/source_only_inception_{}_0_1.pt.tar'.format(dataset)
	baseline_path = 'F:/AI/code_work/phd_codework/chapter4/domain_adaptation/models/normal/mmd/inception_{}_0_1.pt.tar'.format(dataset)
	ffcnn_path = 'F:/AI/code_work/phd_codework/chapter4/domain_adaptation/models/ffcnn/mmd/inception_{}_0_1.pt.tar'.format(dataset)
	dann_path = 'F:/AI/code_work/phd_codework/chapter4/domain_adaptation/models/normal/dann/inception_{}_0_1.pt.tar'.format(dataset)
	darts_path = 'F:/AI/code_work/PHM_NAS_domain_adaptation_new/logs/{}_src_0_seed_{}0/model.pt'.format(dataset.lower(), 30 if dataset == 'CWRU' else 80)
	
	source_only_model = InceptionModel(out_filters=18, classes=classes)#.cuda()
	source_only_model.load_state_dict(torch.load(source_only_path, map_location=torch.device('cpu'))['model_state'])
	
	baseline_model = InceptionModel(out_filters=18, classes=classes)#.cuda()
	baseline_model.load_state_dict(torch.load(baseline_path, map_location=torch.device('cpu'))['model_state'])
	
	ffcnn_model = InceptionTransfer(ff=True, classes=classes)#.cuda()
	ffcnn_model.load_state_dict(torch.load(ffcnn_path, map_location=torch.device('cpu'))['model_state'])
	
	dann_model = InceptionTransfer(dann=True, classes=classes)#.cuda()
	dann_model.load_state_dict(torch.load(dann_path, map_location=torch.device('cpu'))['model_state'])
	
	arch = cwru_src_0_seed_300 if dataset == 'CWRU' else paderborn_src_0_seed_800
	darts_model = Inception(genotypes=arch, C=18, classes=classes)#.cuda()
	darts_model.load_state_dict(torch.load(darts_path, map_location=torch.device('cpu')))
	
	models_list = {'Source_only': source_only_model, 'Baseline': baseline_model, 'FFCNN': ffcnn_model,
	               'DANN': dann_model, 'DARTS': darts_model}
	
	# tsne
	plt.figure(figsize=(10, 6))
	plt.rcParams['font.size'] = 12.
	plt.rcParams['font.family'] = ['Times New Roman']
	marker = ['s', '*', 'o', 'd']
	labels = ['Health', 'Inner race', 'Outer race', 'Ball']
	
	for id, name in enumerate(models_list.keys()):
		plt.subplot(2, 3, id + 1)
		_, fc_src = models_list[name](torch.Tensor(src_x))
		_, fc_tar_1 = models_list[name](torch.Tensor(tar_x_1))
		_, fc_tar_2 = models_list[name](torch.Tensor(tar_x_2))
		_, fc_tar_3 = models_list[name](torch.Tensor(tar_x_3))
		fc_src, fc_tar_1, fc_tar_2, fc_tar_3 = fc_src.detach().cpu().numpy(), fc_tar_1.detach().cpu().numpy(), \
		                   fc_tar_2.detach().cpu().numpy(), fc_tar_3.detach().cpu().numpy()
		features = np.vstack([fc_src, fc_tar_1, fc_tar_2, fc_tar_3])
		
		tsne = TSNE(n_components=2, random_state=args.seed, perplexity=50, learning_rate=500)
		features = tsne.fit_transform(features)
		fc_src, fc_tar_1, fc_tar_2, fc_tar_3 = features[:args.batch_size, :], \
		                                      features[args.batch_size: 2 * args.batch_size, :], \
		                                      features[2 * args.batch_size: 3 * args.batch_size, :], \
		                                      features[3 * args.batch_size:, :]

		
		for i in range(4 if dataset == 'CWRU' else 3):
			plt.scatter(fc_src[src_y == i][:, 0], fc_src[src_y == i][:, 1], marker=marker[i], c='b', edgecolors='b', alpha=0.7, s=30, label='S-{}'.format(labels[i]))
			plt.scatter(fc_tar_1[tar_y_1 == i][:, 0], fc_tar_1[tar_y_1 == i][:, 1], marker=marker[i], c='r', edgecolors='r', alpha=0.7, s=30, label='T-{}'.format(labels[i]))
			plt.scatter(fc_tar_2[tar_y_2 == i][:, 0], fc_tar_2[tar_y_2 == i][:, 1], marker=marker[i], c='g', edgecolors='g', alpha=0.7, s=30, label='T-{}'.format(labels[i]))
			plt.scatter(fc_tar_3[tar_y_3 == i][:, 0], fc_tar_3[tar_y_3 == i][:, 1], marker=marker[i], c='#FFA500', edgecolors='#FFA500', alpha=0.7, s=40, label='T-{}'.format(labels[i]))
		plt.xticks([])
		plt.yticks([])
		plt.title(name)
	
	# 绘制图例
	plt.subplot(2, 3, 6)
	p1, p2, p3, p4 = [], [], [], []
	for i in range(4 if dataset == 'CWRU' else 3):
		t1 = plt.scatter(fc_src[src_y == i][5, 0], fc_src[src_y == i][5, 1], marker=marker[i], c='b', edgecolors='b', alpha=0.7, s=30)
		t2 = plt.scatter(fc_tar_1[tar_y_1 == i][5, 0], fc_tar_1[tar_y_1 == i][5, 1], marker=marker[i], c='r', edgecolors='r', alpha=0.7, s=30)
		t3 = plt.scatter(fc_tar_2[tar_y_2 == i][5, 0], fc_tar_2[tar_y_2 == i][5, 1], marker=marker[i], c='g', edgecolors='g', alpha=0.7, s=30)
		t4 = plt.scatter(fc_tar_3[tar_y_3 == i][5, 0], fc_tar_3[tar_y_3 == i][5, 1], marker=marker[i], c='#FFA500', edgecolors='#FFA500', alpha=0.7, s=40)
		p1.append(t1)
		p2.append(t2)
		p3.append(t3)
		p4.append(t4)
	plt.axvspan(xmin=min(features[:, 0]), xmax=max(features[:, 0]), ymin=min(features[:, 1]), ymax=max(features[:, 1]), facecolor='w')
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')
	plt.legend([tuple(p1), tuple(p2), tuple(p3), tuple(p4)], ['Source', 'Target task 1', 'Target task 2', 'Target task 3'], loc=(0, 0.2), handler_map={tuple: HandlerTuple(ndivide=None)})
	
	plt.tight_layout()
	plt.savefig('./figures/{}_tsne_2.png'.format(dataset.lower()), dpi=600)
	pdf = PdfPages('./figures/{}_tsne_2.pdf'.format(dataset.lower()))
	pdf.savefig()
	pdf.close()


	

if __name__ == '__main__':
	plot_tsne('CWRU')
	plot_tsne('Paderborn')


