'''
Train Neural RE Model
'''
__author__ = 'Maosen'
import os
import random
import torch
import logging
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import utils
from model import Model
from utils import Dataset, DatasetHD, DatasetHDCrossIter

torch.backends.cudnn.deterministic = True


def train(args):
	model = Model(args, device, train_dset.rel2id, word_emb=emb_matrix)
	logging.info('Model: %s, Parameter Number: %d' % (args.model, model.count_parameters()))

	max_dev_f1 = 0.0
	test_result_on_max_dev_f1 = (0.0, 0.0, 0.0)

	for iter in range(niter):
		loss = 0.0

		if args.fix_bias:
			model.set_bias(train_lp)

		for idx, batch in enumerate(tqdm(train_dset.batched_data)):
			loss_batch = model.update(batch)
			loss += loss_batch
		loss /= len(train_dset.batched_data)

		valid_loss, (dev_prec, dev_recall, dev_f1) = model.eval(dev_dset)
		logging.info('Iteration %d, Train loss %f' % (iter, loss))
		logging.info(
			'Dev loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(valid_loss, dev_prec, dev_recall,
																				  dev_f1))

		valid_loss_c, (cdev_prec, cdev_recall, cdev_f1) = model.eval(cdev_dset)
		logging.info(
			'CDev loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(valid_loss_c, cdev_prec, cdev_recall,
																		cdev_f1))



		if args.fix_bias:
			model.set_bias(test_lp)
			logging.warn('Currently test evaluation is using gold test label distribution, only for reference.')

		test_loss, (test_prec, test_recall, test_f1) = model.eval(test_dset)
		logging.info(
			'Test loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(test_loss, test_prec, test_recall,
																				   test_f1))
		if dev_f1 > max_dev_f1:
			max_dev_f1 = dev_f1
			test_result_on_max_dev_f1 = (test_prec, test_recall, test_f1)

			# the saved model should have train_lp as bias.
			if args.fix_bias:
				model.set_bias(train_lp)
			save_filename = os.path.join(args.save_dir, '%s_%d.pkl' % (args.info, runid))
			model.save(save_filename, iter)

		'''
		if (iter+1) in [5, 15, 30]:
			if args.fix_bias:
				model.set_bias(train_lp)
			save_filename = os.path.join(args.save_dir, '%s_%d_e%d.pkl' % (args.info, runid, iter))
			model.save(save_filename, iter)
		'''

		model.update_lr(valid_loss)

	logging.info('Max Dev F1: %.4f' % max_dev_f1)
	test_p, test_r, test_f1 = test_result_on_max_dev_f1
	logging.info('Test P, R, F1 on best epoch: {:.4f}, {:.4f}, {:.4f}'.format(test_p, test_r, test_f1))
	logging.info('\n')

	return max_dev_f1, test_result_on_max_dev_f1

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/neural/KBP', help='specify dataset with directory')
	parser.add_argument('--vocab_dir', type=str, default='data/neural/vocab', help='directory storing word2id file and word embeddings.')

	# Model Specs
	parser.add_argument('--model', type=str, default='bgru', help='model name, (cnn|pcnn|bgru|lstm|palstm)')
	parser.add_argument("--dual", type = bool, default = False, help = "Dual supervision ")
	parser.add_argument("--crossiter", type = bool, default = True)

	parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
	parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
	parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
	parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
	parser.add_argument('--position_dim', type=int, default=30, help='Position encoding dimension.')

	parser.add_argument('--hidden', type=int, default=200, help='RNN hidden state size.')
	parser.add_argument('--window_size', type=int, default=3, help='Convolution window size')
	parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')

	parser.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Bidirectional RNN.' )
	parser.set_defaults(bidirectional=True)
	parser.add_argument('--bias', dest='bias', action='store_true', help='Whether Bias term is used for linear layer.')
	parser.set_defaults(bias=True)
	parser.add_argument('--fix_bias', dest='fix_bias', action='store_true', help='Train model with fix bias (not fixed by default).')
	parser.set_defaults(fix_bias=False)

	parser.add_argument("--multitask", type = bool, default = False)
	parser.add_argument("--union", type=bool, default=False)

	# Data Loading & Pre-processing
	parser.add_argument('--mask_no_type', dest='mask_with_type', action='store_false')
	parser.set_defaults(mask_with_type=True)
	parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
	parser.add_argument('--no-lower', dest='lower', action='store_false')
	parser.set_defaults(lower=False)
	parser.add_argument('--batch_size', type=int, default=64)

	# Optimization
	parser.add_argument("--optimizer", type= str, default="SGD")
	parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
	parser.add_argument('--lr_decay', type=float, default=0.9)
	parser.add_argument("--weight_decay", type=float, default=0.0, help="Applies to Adam")

	parser.add_argument('--num_epoch', type=int, default=30)
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

	# Optimization - Dropout
	parser.add_argument('--in_drop', type=float, default=0.6, help='Input dropout rate.')
	parser.add_argument('--intra_drop', type=float, default=0.1, help='Intra-layer dropout rate.')
	parser.add_argument('--state_drop', type=float, default=0.5, help='RNN state dropout rate.')
	parser.add_argument('--out_drop', type=float, default=0.6, help='Output dropout rate.')

	#Dual Options
	parser.add_argument("--proportion_ha", type = float, default = 0.5, help = "Proportion on human annot")
	parser.add_argument("--use_dev", type = bool, default = False, help = "")
	parser.add_argument("--w_dist", type = float, default = 1e-4, help = "")
	parser.add_argument("--diffall", type = bool, default = False, help = "")
	parser.add_argument("--shuffle", type = bool, default = False)
	parser.add_argument("--haonly", type = bool, default = False)




	# Other options
	parser.add_argument('--seed', type=int, default=7698)
	parser.add_argument('--repeat', type=int, default=5, help='train the model for multiple times.')
	parser.add_argument('--save_dir', type=str, default='dumped_models', help='Root dir for saving models.')
	parser.add_argument('--info', type=str, default='KBP_default', help='description, also used as filename to save model.')
	parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
	parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
	parser.add_argument("--skip_ifexists",type =bool, default = False, help = "Skip experiment if exists" )



	args = parser.parse_args()

	# Set random seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	# Load vocab file (id2word)
	with open(args.vocab_dir + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	word2id = {}
	for idx, word in enumerate(vocab):
		word2id[word] = idx

	# Load word embedding
	emb_file = args.vocab_dir + '/embedding.npy'
	emb_matrix = np.load(emb_file)
	assert emb_matrix.shape[0] == len(vocab)
	assert emb_matrix.shape[1] == args.emb_dim
	args.vocab_size = len(vocab)
	niter = args.num_epoch

	if args.cpu:
		args.cuda = False
	device = torch.device("cuda:0" if args.cuda else "cpu")
	print('Using device: %s' % device.type)

	# Load data.
	print('Reading data......')
	rel2id = utils.load_rel2id('%s/relation2id.json' % args.data_dir)
	train_filename = '%s/train.json' % args.data_dir
	test_filename = '%s/test.json' % args.data_dir
	dev_filename = '%s/dev.json' % args.data_dir




	train_ds_dset = Dataset(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True,
							mask_with_type=args.mask_with_type)
	dev_dset = Dataset(dev_filename, args, word2id, device, rel2id=rel2id, mask_with_type=args.mask_with_type)
	cdev_dset, test_dset = utils.get_cv_dataset(test_filename, args, word2id, device, rel2id, dev_ratio=0.2)


	if args.dual or args.multitask or args.union:
		if args.crossiter:
			train_dset = DatasetHDCrossIter(train_ds_dset, cdev_dset, dev_dset, args, word2id, device, use_dev = args.use_dev, mask_with_type=args.mask_with_type)
		else:
			train_dset = utils.build_hd_dataset_shuffle(train_ds_dset, cdev_dset, args, word2id, device, rel2id = rel2id)
	elif args.haonly:
		train_dset = cdev_dset
	else:
		train_dset = train_ds_dset
		# Get label distribution from train set. Used in fix_bias.
		train_lp = torch.from_numpy(train_dset.log_prior).to(device)
		#DEBUG by XXXXXXX
		#test_lp = torch.from_numpy(test_dset.log_prior).to(device)
		test_lp = torch.from_numpy(cdev_dset.log_prior)

	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)

	info_list = []

	for fname in os.listdir(args.save_dir):
		info_list.append(fname[:-4])

	for runid in range(1, args.repeat + 1):
		logging.info('Run model #%d time......' % runid)
		info_name = f"{args.info}_{runid}"
		if args.skip_ifexists and info_name in info_list:
			print(f"Skip {info_name}")
			continue
		dev_f1, test_result = train(args)
		logging.info('')
