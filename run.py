import os


seed = 800

search_commands = [
	'nohup python -u train_search.py --dataset CWRU --src 0 --seed {} --save cwru_src_0_seed_{} --gpu 0 > ./logs/cwru_src_0_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset CWRU --src 1 --seed {} --save cwru_src_1_seed_{} --gpu 1 > ./logs/cwru_src_1_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset CWRU --src 2 --seed {} --save cwru_src_2_seed_{} --gpu 2 > ./logs/cwru_src_2_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset CWRU --src 3 --seed {} --save cwru_src_3_seed_{} --gpu 3 > ./logs/cwru_src_3_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset Paderborn --src 0 --seed {} --save paderborn_src_0_seed_{} --gpu 4 > ./logs/paderborn_src_0_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset Paderborn --src 1 --seed {} --save paderborn_src_1_seed_{} --gpu 5 > ./logs/paderborn_src_1_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset Paderborn --src 2 --seed {} --save paderborn_src_2_seed_{} --gpu 6 > ./logs/paderborn_src_2_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train_search.py --dataset Paderborn --src 3 --seed {} --save paderborn_src_3_seed_{} --gpu 7 > ./logs/paderborn_src_3_seed_{}.log 2>&1 &'.format(seed, seed, seed)
]

train_commands = [
	'nohup python -u train.py --dataset CWRU --src 0 --seed {} --arch cwru_src_0_seed_{} --gpu 0 > ./logs/train_cwru_src_0_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset CWRU --src 1 --seed {} --arch cwru_src_1_seed_{} --gpu 1 > ./logs/train_cwru_src_1_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset CWRU --src 2 --seed {} --arch cwru_src_2_seed_{} --gpu 2 > ./logs/train_cwru_src_2_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset CWRU --src 3 --seed {} --arch cwru_src_3_seed_{} --gpu 3 > ./logs/train_cwru_src_3_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset Paderborn --src 0 --seed {} --arch paderborn_src_0_seed_{} --gpu 4 > ./logs/train_paderborn_src_0_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset Paderborn --src 1 --seed {} --arch paderborn_src_1_seed_{} --gpu 5 > ./logs/train_paderborn_src_1_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset Paderborn --src 2 --seed {} --arch paderborn_src_2_seed_{} --gpu 6 > ./logs/train_paderborn_src_2_seed_{}.log 2>&1 &'.format(seed, seed, seed),
	'nohup python -u train.py --dataset Paderborn --src 3 --seed {} --arch paderborn_src_3_seed_{} --gpu 7 > ./logs/train_paderborn_src_3_seed_{}.log 2>&1 &'.format(seed, seed, seed)
]

train_commands_wo_a = [
	'nohup python -u train.py --dataset CWRU --src 0 --seed 100 --arch cwru_src_0_seed_100_wo_a --gpu 0 > ./logs/train_cwru_src_0_seed_100_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 300 --arch cwru_src_0_seed_300_wo_a --gpu 1 > ./logs/train_cwru_src_0_seed_300_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 400 --arch cwru_src_0_seed_400_wo_a --gpu 2 > ./logs/train_cwru_src_0_seed_400_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 500 --arch cwru_src_0_seed_500_wo_a --gpu 3 > ./logs/train_cwru_src_0_seed_500_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 800 --arch cwru_src_0_seed_800_wo_a --gpu 3 > ./logs/train_cwru_src_0_seed_800_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 100 --arch paderborn_src_0_seed_100_wo_a --gpu 4 > ./logs/train_paderborn_src_0_seed_100_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 300 --arch paderborn_src_0_seed_300_wo_a --gpu 5 > ./logs/train_paderborn_src_0_seed_300_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 400 --arch paderborn_src_0_seed_400_wo_a --gpu 6 > ./logs/train_paderborn_src_0_seed_400_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 500 --arch paderborn_src_0_seed_500_wo_a --gpu 7 > ./logs/train_paderborn_src_0_seed_500_wo_a.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 800 --arch paderborn_src_0_seed_800_wo_a --gpu 7 > ./logs/train_paderborn_src_0_seed_800_wo_a.log 2>&1 &',
]

train_commands_wo_f = [
	'nohup python -u train.py --dataset CWRU --src 0 --seed 100 --arch cwru_src_0_seed_100_wo_f --gpu 0 > ./logs/train_cwru_src_0_seed_100_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 300 --arch cwru_src_0_seed_300_wo_f --gpu 1 > ./logs/train_cwru_src_0_seed_300_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 400 --arch cwru_src_0_seed_400_wo_f --gpu 2 > ./logs/train_cwru_src_0_seed_400_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 500 --arch cwru_src_0_seed_500_wo_f --gpu 3 > ./logs/train_cwru_src_0_seed_500_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset CWRU --src 0 --seed 800 --arch cwru_src_0_seed_800_wo_f --gpu 0 > ./logs/train_cwru_src_0_seed_800_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 100 --arch paderborn_src_0_seed_100_wo_f --gpu 4 > ./logs/train_paderborn_src_0_seed_100_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 300 --arch paderborn_src_0_seed_300_wo_f --gpu 5 > ./logs/train_paderborn_src_0_seed_300_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 400 --arch paderborn_src_0_seed_400_wo_f --gpu 6 > ./logs/train_paderborn_src_0_seed_400_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 500 --arch paderborn_src_0_seed_500_wo_f --gpu 7 > ./logs/train_paderborn_src_0_seed_500_wo_f.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 800 --arch paderborn_src_0_seed_800_wo_f --gpu 4 > ./logs/train_paderborn_src_0_seed_800_wo_f.log 2>&1 &',
]


train_commands_random = [
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 10 --arch random_0 --gpu 0 > ./logs/random_0.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 20 --arch random_1 --gpu 1 > ./logs/random_1.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 30 --arch random_2 --gpu 2 > ./logs/random_2.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 40 --arch random_3 --gpu 3 > ./logs/random_3.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 50 --arch random_4 --gpu 3 > ./logs/random_4.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 60 --arch random_5 --gpu 4 > ./logs/random_5.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 70 --arch random_6 --gpu 5 > ./logs/random_6.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 80 --arch random_7 --gpu 6 > ./logs/random_7.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 90 --arch random_8 --gpu 7 > ./logs/random_8.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 100 --arch random_9 --gpu 3 > ./logs/random_9.log 2>&1 &',
]


train_commands_epoch = [
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 10 --arch epoch_10_cwru --gpu 0 > ./logs/epoch_10_cwru.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 20 --arch epoch_20_cwru --gpu 1 > ./logs/epoch_20_cwru.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 30 --arch epoch_30_cwru --gpu 3 > ./logs/epoch_30_cwru.log 2>&1 &',
	# 'nohup python -u train.py --dataset CWRU --src 0 --seed 40 --arch epoch_40_cwru --gpu 5 > ./logs/epoch_40_cwru.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 50 --arch epoch_10_paderborn --gpu 4 > ./logs/epoch_10_paderborn.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 60 --arch epoch_20_paderborn --gpu 5 > ./logs/epoch_20_paderborn.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 70 --arch epoch_30_paderborn --gpu 6 > ./logs/epoch_30_paderborn.log 2>&1 &',
	'nohup python -u train.py --dataset Paderborn --src 0 --seed 80 --arch epoch_40_paderborn --gpu 7 > ./logs/epoch_40_paderborn.log 2>&1 &',
]


def run_search():
	for c in search_commands:
		p = os.system(c)
		print(c)
		print(p)


def run_train():
	for c in train_commands:
		p = os.system(c)
		print(c)
		print(p)
		

def run_train_wo_a():
	for c in train_commands_wo_a:
		p = os.system(c)
		print(c)
		print(p)


def run_train_wo_f():
	for c in train_commands_wo_f:
		p = os.system(c)
		print(c)
		print(p)


def run_train_random():
	for c in train_commands_random:
		p = os.system(c)
		print(c)
		print(p)


def run_train_epoch():
	for c in train_commands_epoch:
		p = os.system(c)
		print(c)
		print(p)



if __name__ == '__main__':
	# run_search()
	# run_train()
	# run_train_wo_a()
	# run_train_wo_f()
	run_train_random()
	# run_train_epoch()
	

