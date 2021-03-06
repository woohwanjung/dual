'''
Training script for Position-Aware LSTM for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
from model import Model
import utils
from utils import Dataset, DatasetBERT
import argparse
import pickle
import numpy as np
import os
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='dumped_models', help='Root dir for saving models.')
parser.add_argument('--eval_result_dir', type=str, default='eval_result', help='Root dir for saving results.')
parser.add_argument('--info', type=str, default='KBP_default', help='description, also used as filename to save model.')
parser.add_argument('--repeat', type=int, default=5, help='test the model for multiple trains.')
# if info == 'KBP_default' and repeat == 5, we will evaluate 5 models 'KBP_default_1' ... 'KBP_default_5'

parser.add_argument('--thres_ratio', type=float, default=0.2, help='proportion of data to tune thres.')
parser.add_argument('--bias_ratio', type=float, default=0.2, help='proportion of data to estimate bias.')
parser.add_argument('--cvnum', type=int, default=100, help='# samples to tune thres or estimate bias')

parser.add_argument('--fix_bias', dest='fix_bias', action='store_true', help='test model with fix bias (not fixed by default).')
parser.set_defaults(fix_bias=False)

#Dual Options
parser.add_argument("--multitask", type = bool, default = False)
parser.add_argument("--proportion_ha", type = float, default = 0.5, help = "Proportion on human annot")
parser.add_argument("--use_dev", type = bool, default = False, help = "")
parser.add_argument("--w_dist", type = float, default = 1e-4, help = "")
parser.add_argument("--diffall", type = bool, default = False, help = "")



args_new = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the config of trained model
model_file = os.path.join(args_new.save_dir, args_new.info)
params = torch.load(model_file + '_1.pkl')
args = params['config']
print(args)

attrs_w_default_vals =[("multitask", False), ("dual", False), ("optimizer", "SGD")]
for attr, d_val in attrs_w_default_vals:
	if not hasattr(args, attr):
		setattr(args, attr, d_val)

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

with open(args.vocab_dir + '/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
word2id = {}
for idx, word in enumerate(vocab):
	word2id[word] = idx

emb_file = args.vocab_dir + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(vocab)
assert emb_matrix.shape[1] == args.emb_dim
args.vocab_size = len(vocab)


rel2id = utils.load_rel2id('%s/relation2id.json' % args.data_dir)
none_id = rel2id['no_relation']
print('Reading data......')
train_filename = '%s/train.json' % args.data_dir
test_filename = '%s/test.json' % args.data_dir
dev_filename = '%s/dev.json' % args.data_dir




train_dset = DatasetBERT(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True)
dev_dset = DatasetBERT(dev_filename, args, word2id, device, rel2id=rel2id)
cdev_dset, test_dset = utils.get_cv_dataset_bert(test_filename, args, word2id, device, rel2id, dev_ratio=0.2)
train_lp = torch.from_numpy(train_dset.log_prior).to(device)


if "dual" in args_new.info:
	eval_result_dual = utils.ExpResult(args_new.info)
elif not args_new.fix_bias:
	eval_result_original = utils.ExpResult(args_new.info,"original")
	eval_result_maxth = utils.ExpResult(args_new.info,"max_th")
	eval_result_entth = utils.ExpResult(args_new.info,"ent_th")
	eval_result_setb = utils.ExpResult(args_new.info,"setb")
else:
	eval_result_fixb = utils.ExpResult(args_new.info,"fixb")


for runid in range(1, args_new.repeat + 1):
	model = Model(args, device, word_emb=emb_matrix, rel2id=rel2id)
	print('loading model %d ......' % runid)
	model_path = '%s_%d.pkl' % (model_file, runid)
	if not os.path.exists(model_path):
		continue
	model.load(model_path)

	if "dual" in args_new.info:
		test_loss, (prec, recall, f1) = model.eval(test_dset, col_label=3, col_origidx=4)
		print('Original:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))
		eval_result_dual.add_result(runid, (prec, recall, f1))
	elif not args_new.fix_bias:
		print('Evaluating original / max_thres / ent_thres / set_bias.')

		# Original
		test_loss, (prec, recall, f1) = model.eval(test_dset, col_label=3, col_origidx=4)
		print('Original:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))
		eval_result_original.add_result(runid, (prec, recall, f1))

		# Max Thres
		test_loss, f1, recall, prec = model.TuneMaxThres(test_dset, none_id,
																  ratio=args_new.thres_ratio,
																  cvnum=args_new.cvnum,
														 col_label=3, col_origidx=4)
		print('Max Thres:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))
		eval_result_maxth.add_result(runid, (prec, recall, f1))
		# Entropy Thres
		test_loss, f1, recall, prec = model.TuneEntropyThres(test_dset, none_id,
																	  ratio=args_new.thres_ratio,
																	  cvnum=args_new.cvnum, col_label=3, col_origidx=4)
		print('Entropy Thres:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))
		eval_result_entth.add_result(runid, (prec, recall, f1))

		# Set bias
		#results = []
		#for j in tqdm(range(args_new.cvnum)):
			# splitting test set into clean dev and actual test
			#cdev_dset, test_dset = utils.get_cv_dataset(test_filename, args, word2id, device, rel2id, dev_ratio=args_new.thres_ratio)
		cdev_lp = torch.from_numpy(cdev_dset.log_prior).to(device)
		bias_old = model.get_bias()
		bias_new = bias_old - train_lp + cdev_lp
		model.set_bias(bias_new)
		test_loss, (prec, recall, f1) = model.eval(test_dset, col_label=3, col_origidx=4)
		#results.append((test_loss, prec, recall, f1))
		model.set_bias(bias_old)

		#results = np.array(results, dtype=np.float32)
		#test_loss, prec, recall, f1 = np.mean(results, axis=0)
		print('Set bias:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))
		eval_result_setb.add_result(runid, (prec, recall, f1))
	else:
		#results = []
		#for j in tqdm(range(args_new.cvnum)):
			# splitting test set into clean dev and actual test
			#cdev_dset, test_dset = utils.get_cv_dataset(test_filename, args, word2id, device, rel2id, dev_ratio=args_new.thres_ratio)
		cdev_lp = torch.from_numpy(cdev_dset.log_prior).to(device)
		bias_old = model.get_bias()
		bias_new = cdev_lp

		print(bias_old)
		print(bias_new)
		model.set_bias(bias_new)
		test_loss, (prec, recall, f1) = model.eval(test_dset, col_label=3, col_origidx=4)
		#results.append((test_loss, prec, recall, f1))
		model.set_bias(bias_old)

		#results = np.array(results, dtype=np.float32)
		#test_loss, prec, recall, f1 = np.mean(results, axis=0)
		print('Fix bias:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))
		eval_result_fixb.add_result(runid, (prec, recall, f1))


if "dual" in args_new.info:
	if eval_result_dual.num_exp() > 0:
		eval_result_dual.save(args_new.eval_result_dir)
		eval_result_dual.print_result()
elif not args_new.fix_bias:
	if eval_result_original.num_exp() >0:
		eval_result_original.save(args_new.eval_result_dir)
		eval_result_maxth.save(args_new.eval_result_dir)
		eval_result_entth.save(args_new.eval_result_dir)
		eval_result_setb.save(args_new.eval_result_dir)

		eval_result_original.print_result()
		eval_result_maxth.print_result()
		eval_result_entth.print_result()
		eval_result_setb.print_result()
else:
	if eval_result_fixb.num_exp()>0:
		eval_result_fixb.save(args_new.eval_result_dir)
		eval_result_fixb.print_result()
