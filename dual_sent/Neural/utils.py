'''
Data Loader for Position-Aware LSTM for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import json
import math
import random
import pickle as pkl

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

ner2id = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NATIONALITY': 2, 'SET': 3, 'ORDINAL': 4, 'ORGANIZATION': 5, 'MONEY': 6, 'PERCENT': 7, 'URL': 8, 'DURATION': 9, 'PERSON': 10, 'CITY': 11, 'CRIMINAL_CHARGE': 12, 'DATE': 13, 'TIME': 14, 'NUMBER': 15, 'STATE_OR_PROVINCE': 16, 'RELIGION': 17, 'MISC': 18, 'CAUSE_OF_DEATH': 19, 'LOCATION': 20, 'TITLE': 21, 'O': 22, 'COUNTRY': 23, 'IDEOLOGY': 24}
pos2id = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}
NO_RELATION = 0

MAXLEN = 300
not_existed = {}

import os
def log2prob(log_prob):
	exps = np.exp(log_prob)
	return exps / np.sum(exps)

def log2posprob(log_prob):
	log_prob = log_prob[1:]
	exps = np.exp(log_prob)
	return exps / np.sum(exps)

def load_rel2id(fname):
	with open(fname, 'r') as f:
		relation2id = json.load(f)
		return relation2id

def ensure_dir(d, verbose=True):
	if not os.path.exists(d):
		if verbose:
			print("Directory {} do not exist; creating...".format(d))
		os.makedirs(d)
def stat_tensor(t):
	print(f"Mean: {t.mean()}")
	print(f"[{t.min()},{t.max()}]")
	print(t.shape)


class ExpResult(object):
	attrs = ["Precision", "Recall", "F1"]
	def __init__(self, info, opt = None):
		self.info = info
		self.opt = opt
		self.results = {}
	def num_exp(self):
		return len(self.results)
	def add_result(self, runid, res):
		self.results[runid] = res

	def get_aggregated_result(self):
		num_result = len(self.results)
		aggregated_result = np.zeros((num_result, len(self.attrs)))
		for i, res in enumerate(self.results.values()):
			for a in range(len(self.attrs)):
				aggregated_result[i,a] = res[a]
		mean = aggregated_result.mean(axis = 0)
		std = aggregated_result.std(axis = 0)
		max_val = aggregated_result.max(axis = 0)
		min_val = aggregated_result.min(axis = 0)

		return mean, std, max_val, min_val, num_result

	def save(self, dirpath):
		filepath = self.filepath(dirpath, self.info, self.opt)

		with open(filepath, "wb") as f:
			pkl.dump(self, f)

	@classmethod
	def filepath(cls, dirpath, info, opt= None):
		if opt is None:
			filepath = f"{dirpath}/{info}.pkl"
		else:
			filepath = f"{dirpath}/{info}_{opt}.pkl"
		return filepath


	@classmethod
	def load(cls, dirpath, info, opt = None):
		filepath = cls.filepath(dirpath, info, opt)
		with open(filepath, "rb") as f:
			res = pkl.load(f)
			return res

	def print_result(self):
		print(self.info, self.opt)
		mean, std, max_val, min_val, num_result = self.get_aggregated_result()
		head = "\t".join(self.attrs)
		print(head)
		val_str = ""
		for a in range(len(self.attrs)):
			val_str += f"{mean[a]*100:.2f} ({std[a]*100:.2f})\t"
		print(val_str)

		val_str = ""
		for a in range(len(self.attrs)):
			val_str = f"[{min_val[a]*100:.2f}, {max_val[a]*100:.2f}]"
		print(val_str)







class Dataset(object):
	def __init__(self, filename, args, word2id, device, rel2id=None, shuffle=False, batch_size=None, mask_with_type=True, use_mask=True, verbose=True, instance_only = False):
		if batch_size is None:
			batch_size = args.batch_size

		lower = args.lower
		self.device = device
		instances = self.load_instances(filename)
		self.instances = instances

		self.set_rel2id(rel2id, instances)
		rel2id = self.rel2id

		datasize = len(instances)
		if shuffle:
			indices = list(range(datasize))
			random.shuffle(indices)
			instances = [instances[i] for i in indices]

		data = []
		labels = []
		discard = 0
		rel_cnt = {}
		# preprocess: convert tokens to id
		for instance in instances:
			tokens = instance['token']
			l = len(tokens)
			if l > MAXLEN or l != len(instance['stanford_ner']):
				discard += 1
				continue
			if lower:
				tokens = [t.lower() for t in tokens]
			# anonymize tokens
			ss, se = instance['subj_start'], instance['subj_end']
			os, oe = instance['obj_start'], instance['obj_end']
			# replace subject and object with typed "placeholder"
			if use_mask:
				if mask_with_type:
					tokens[ss:se + 1] = ['SUBJ-' + instance['subj_type']] * (se - ss + 1)
					tokens[os:oe + 1] = ['OBJ-' + instance['obj_type']] * (oe - os + 1)
				else:
					tokens[ss:se + 1] = ['SUBJ-O'] * (se - ss + 1)
					tokens[os:oe + 1] = ['OBJ-O'] * (oe - os + 1)
			tokens = map_to_ids(tokens, word2id)
			pos = map_to_ids(instance['stanford_pos'], pos2id)
			ner = map_to_ids(instance['stanford_ner'], ner2id)
			subj_positions = get_positions(ss, se, l)
			obj_positions = get_positions(os, oe, l)
			# if instance['relation'] in self.rel2id and instance['relation'] != 'per:countries_of_residence':
			if instance['relation'] in self.rel2id:
				rel = instance['relation']
				relation = self.rel2id[instance['relation']]
			else:
				# relation = self.rel2id['no_relation']
				discard += 1
				continue
			if rel not in rel_cnt:
				rel_cnt[rel] = 0
			rel_cnt[rel] += 1
			data.append((tokens, pos, ner, subj_positions, obj_positions, relation))
			labels.append(relation)

		datasize = len(data)
		self.datasize = datasize
		self.rel_cnt = rel_cnt
		self.data = data
		self.labels = labels

		if instance_only:
			return

		self.log_prior = np.zeros(len(rel2id), dtype=np.float32)
		self.rel_distrib = np.zeros(len(rel2id), dtype=np.float32)
		for rel in rel_cnt:
			relid = rel2id[rel]
			self.rel_distrib[relid] = rel_cnt[rel]
			self.log_prior[relid] = np.log(rel_cnt[rel])
		max_log = np.max(self.log_prior)
		self.log_prior = self.log_prior - max_log
		self.rel_distrib = self.rel_distrib / np.sum(self.rel_distrib)

		# chunk into batches
		batched_data = [data[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_label = [labels[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_data = []
		for batch in batched_data:
			batch_size = len(batch)
			batch = list(zip(*batch))
			assert len(batch) == 6
			# sort by descending order of lens
			lens = [len(x) for x in batch[0]]
			batch, orig_idx = sort_all(batch, lens)

			words = get_padded_tensor(batch[0], batch_size)
			pos = get_padded_tensor(batch[1], batch_size)
			ner = get_padded_tensor(batch[2], batch_size)
			subj_pos = get_padded_tensor(batch[3], batch_size)
			obj_pos = get_padded_tensor(batch[4], batch_size)
			relations = torch.tensor(batch[5], dtype=torch.long)
			self.batched_data.append((words, pos, ner, subj_pos, obj_pos, relations, orig_idx))

		if verbose:
			print('Discard instances: %d, Total instances: %d, Number of batches: %d' % (discard, datasize, len(batched_data)))

	def get_id_maps(self, instances):
		print('Getting index maps......')
		self.rel2id = {}
		rel_set = ['no_relation']
		for instance in tqdm(instances):
			rel = instance['relation']
			if rel not in rel_set:
				rel_set.append(rel)
		for idx, rel in enumerate(rel_set):
			self.rel2id[rel] = idx
		NO_RELATION = self.rel2id['no_relation']
		print(self.rel2id)

	def load_instances(self, filename):
		if isinstance(filename, str):
			with open(filename, 'r') as f:
				instances = json.load(f)
		else:
			instances = filename
		return instances

	def set_rel2id(self, rel2id, instances):
		if rel2id == None:
			self.get_id_maps(instances)
			rel2id = self.rel2id
		else:
			self.rel2id = rel2id

class DatasetBERT(Dataset):
	def __init__(self, filename, args, word2id, device, instances= None, rel2id=None, shuffle=False, batch_size=None, verbose=True, instance_only=False, postfix = ""):
		if batch_size is None:
			batch_size = args.batch_size

		self.device = device
		if instances is None:
			instances = self.load_instances(filename)
		self.instances = instances

		self.set_rel2id(rel2id, instances)
		rel2id = self.rel2id

		data, labels, rel_cnt = self.load_data(filename, shuffle, args.seed, postfix = postfix)

		datasize = len(data)
		self.datasize = datasize
		self.rel_cnt = rel_cnt
		self.data = data
		self.labels = labels

		if instance_only:
			return

		self.log_prior = np.zeros(len(rel2id), dtype=np.float32)
		self.rel_distrib = np.zeros(len(rel2id), dtype=np.float32)
		for rel in rel_cnt:
			relid = rel2id[rel]
			self.rel_distrib[relid] = rel_cnt[rel]
			self.log_prior[relid] = np.log(rel_cnt[rel])
		max_log = np.max(self.log_prior)
		self.log_prior = self.log_prior - max_log
		self.rel_distrib = self.rel_distrib / np.sum(self.rel_distrib)

		# chunk into batches
		batched_data = [data[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_label = [labels[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_data = []
		for batch in batched_data:
			batch_size = len(batch)
			batch = list(zip(*batch))
			assert len(batch) == 3
			# sort by descending order of lens
			lens = [len(x) for x in batch[0]]
			batch, orig_idx = sort_all(batch, lens)

			words = get_padded_tensor(batch[0], batch_size)
			#subj_map = get_select_map(batcg)
			#subj_pos = get_padded_tensor(batch[1][0], batch_size)
			#obj_pos = get_padded_tensor(batch[1][1], batch_size)
			#subj_pos, obj_pos = batch[1]
			sbj_map = torch.zeros(words.shape, device=words.device)
			obj_map = torch.zeros(words.shape, device=words.device)
			for bi, (range_sbj, range_obj) in enumerate(batch[1]):
				weight_sbj = 1.0/(range_sbj[1]-range_sbj[0])
				weight_obj = 1.0/(range_obj[1]-range_obj[0])
				sbj_map[bi,range_sbj[0]:range_sbj[1]] = weight_sbj
				obj_map[bi,range_obj[0]:range_obj[1]] = weight_obj

			relations = torch.tensor(batch[2], dtype=torch.long)
			self.batched_data.append((words, sbj_map, obj_map, relations, orig_idx))

		if verbose:
			print('Total instances: %d, Number of batches: %d' % (datasize, len(batched_data)))

	def load_data(self, filename, shuffle, seed, postfix = ""):
		instances =self.instances

		fname_data = filename.replace(".json","_bert.json")
		if shuffle:
			fname_data = fname_data.replace("_bert.json", f"_bert{seed}.json")
		fname_data = fname_data.replace(".json",f"{postfix}.json")
		#import os
		if os.path.exists(fname_data):
			with open(fname_data, "r") as f:
				dat_dict = json.load(f)
			return dat_dict['data'], dat_dict['labels'], dat_dict['rel_cnt']

		dat_dict = {}
		datasize = len(instances)
		if shuffle:
			indices = list(range(datasize))
			random.shuffle(indices)
			instances = [instances[i] for i in indices]

		data = []
		labels = []
		discard = 0
		rel_cnt = {}
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		for instance in tqdm(instances):
			tokens = instance['token']
			l = len(tokens)
			if l > MAXLEN or l != len(instance['stanford_ner']):
				discard += 1
				continue
			tokens = [t.lower() for t in tokens]
			# anonymize tokens
			# ss, se = instance['subj_start'], instance['subj_end']
			# os, oe = instance['obj_start'], instance['obj_end']
			subwords = list(map(tokenizer.tokenize, tokens))

			starts = [0]
			flatten_subwords = []
			for sw in subwords:
				starts.append(starts[-1] + len(sw))
				flatten_subwords += sw
			#flatten_subwords.append(102)
			subwords_ids = [101]+tokenizer.convert_tokens_to_ids(flatten_subwords)+[102]

			# +1 due to the special token [cls]
			sbj_b = starts[instance['subj_start']] + 1
			sbj_e = starts[instance['subj_end'] + 1] + 1

			obj_b = starts[instance['obj_start']] + 1
			obj_e = starts[instance['obj_end'] + 1] + 1
			entity_range = ((sbj_b, sbj_e), (obj_b, obj_e))
			if instance['relation'] in self.rel2id:
				rel = instance['relation']
				relation = self.rel2id[instance['relation']]
			else:
				# relation = self.rel2id['no_relation']
				discard += 1
				continue

			if rel not in rel_cnt:
				rel_cnt[rel] = 0
			rel_cnt[rel] += 1
			data.append((subwords_ids, entity_range, relation))
			labels.append(relation)

			#if len(data) == 1000:
			#	break

		dat_dict['data'] = data
		dat_dict['labels'] = labels
		dat_dict['rel_cnt'] = rel_cnt
		with open(fname_data, "w") as f:
			json.dump(dat_dict, f)
		return data, labels, rel_cnt

class DatasetHD(object):
	def __init__(self, ds_data, ha_data, args, word2id, device, rel2id=None, batch_size=None, mask_with_type=True, use_mask=True, verbose=True, use_dev = False):
		self.device = device
		self.rel2id = rel2id
		batch_size = args.batch_size
		batch_size_ha = batch_size//2
		batch_size_ds = batch_size - batch_size_ha

		instances = []
		data = []
		labels = []

		id_ha = 0
		id_ds = 0
		for bid_ds in range(ds_data.datasize//batch_size_ds):
			for _ in range(batch_size_ds):
				instances.append(ds_data.instances[id_ds])
				#tokens, pos, ner, subj_positions, obj_positions, relation = ds_data.data[id_ds]
				#data.append((tokens, pos, ner, subj_positions, obj_positions, relation, False))
				data.append(ds_data.data[id_ds]+ (False,))
				labels.append(ds_data.labels[id_ds])
				id_ds += 1

			for _ in range(batch_size_ha):
				instances.append(ha_data.instances[id_ha])
				#tokens, pos, ner, subj_positions, obj_positions, relation = ha_data.data[id_ha]
				#data.append((tokens, pos, ner, subj_positions, obj_positions, relation, True))
				data.append(ha_data.data[id_ha] + (True,))
				labels.append(ha_data.labels[id_ha])
				id_ha += 1
				id_ha %= ha_data.datasize

		datasize = len(instances)

		# chunk into batches
		batched_data = [data[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_label = [labels[i:i + batch_size] for i in range(0, datasize, batch_size)]
		self.batched_data = []
		for batch in batched_data:
			batch_size = len(batch)
			batch = list(zip(*batch))
			if len(batch) == 7:
				# sort by descending order of lens
				lens = [len(x) for x in batch[0]]
				batch, orig_idx = sort_all(batch, lens)

				words = get_padded_tensor(batch[0], batch_size)
				pos = get_padded_tensor(batch[1], batch_size)
				ner = get_padded_tensor(batch[2], batch_size)
				subj_pos = get_padded_tensor(batch[3], batch_size)
				obj_pos = get_padded_tensor(batch[4], batch_size)
				relations = torch.tensor(batch[5], dtype=torch.long)
				is_ha = torch.tensor(batch[6], dtype=torch.float)
				self.batched_data.append((words, pos, ner, subj_pos, obj_pos, relations, orig_idx, is_ha))




class DatasetHDCrossIter(object):
	def __init__(self, ds_data, ha_data_cdev, ha_data_dev, args, word2id, device, rel2id=None, batch_size=None, mask_with_type=True, use_mask=True, verbose=True, use_dev = False):
		self.device = ds_data.device
		self.rel2id = ds_data.rel2id
		batch_size = args.batch_size
		#batch_size_ha = int(args.proportion_ha * batch_size)
		#batch_size_ds = batch_size - batch_size_ha

		#data_size_ha = ha_data.datasize
		batched_data = []
		batched_label = []
		bi_ha_cd = 0
		bi_ha_d = 0
		for bi in range(len(ds_data.batched_data)):
			ha = False
			#words, pos, ner, subj_pos, obj_pos, relations, orig_idx = ds_data.batched_data[bi]
			batched_data.append(ds_data.batched_data[bi] +  (ha,))
			batched_label.append(ds_data.batched_label[bi])


			bi_ha_cd += 1
			bi_ha_cd = bi_ha_cd%len(ha_data_cdev.batched_data)
			ha = True
			#words, pos, ner, subj_pos, obj_pos, relations, orig_idx = ha_data_cdev.batched_data[bi_ha_cd]
			#batched_data.append((words, pos, ner, subj_pos, obj_pos, relations, orig_idx, ha))
			batched_data.append(ha_data_cdev.batched_data[bi_ha_cd] + (ha,))
			batched_label.append(ha_data_cdev.batched_label[bi_ha_cd])


		self.batched_data = batched_data
		self.batched_label = batched_label



def get_padded_tensor(tokens_list, batch_size):
	""" Convert tokens list to a padded Tensor. """
	token_len = max(len(x) for x in tokens_list)
	pad_len = min(token_len, MAXLEN)
	tokens = torch.zeros(batch_size, pad_len, dtype=torch.long).fill_(PAD_ID)
	for i, s in enumerate(tokens_list):
		cur_len = min(pad_len, len(s))
		tokens[i, :cur_len] = torch.tensor(s[:cur_len], dtype=torch.long)
	return tokens

def get_select_map(idx_list, map_size):
	smap = torch.zeros(map_size)
	for i, (b, e) in enumerate(idx_list):
		range_len = e-b
		smap[i][b:e] = 1.0/range_len
	return smap

def split_cv_dataset(filename, args, word2id, device, rel2id, dev_ratio=0.2,instance_only = False):
	with open(filename, 'r') as f:
		instances = json.load(f)

	datasize = len(instances)
	dev_cnt = math.ceil(datasize * dev_ratio)

	indices = list(range(datasize))
	random.shuffle(indices)
	instances = [instances[i] for i in indices]

	dev_instances = instances[:dev_cnt]
	test_instances = instances[dev_cnt:]

	dev_dset = Dataset(dev_instances, args, word2id, device, rel2id, verbose=False, instance_only = instance_only)
	test_dset = Dataset(test_instances, args, word2id, device, rel2id, verbose=False, instance_only = instance_only)
	return dev_dset, test_dset


def get_cv_dataset(filename, args, word2id, device, rel2id, dev_ratio=0.2,instance_only = False):
	with open(filename, 'r') as f:
		instances = json.load(f)

	datasize = len(instances)
	dev_cnt = math.ceil(datasize * dev_ratio)

	indices = list(range(datasize))
	random.shuffle(indices)
	instances = [instances[i] for i in indices]

	dev_instances = instances[:dev_cnt]
	test_instances = instances[dev_cnt:]

	dev_dset = Dataset(dev_instances, args, word2id, device, rel2id, verbose=False, instance_only = instance_only)
	test_dset = Dataset(test_instances, args, word2id, device, rel2id, verbose=False, instance_only = instance_only)
	return dev_dset, test_dset

def get_cv_dataset_bert(filename, args, word2id, device, rel2id, dev_ratio=0.2,instance_only = False):
	with open(filename, 'r') as f:
		instances = json.load(f)

	datasize = len(instances)
	dev_cnt = math.ceil(datasize * dev_ratio)

	indices = list(range(datasize))
	random.shuffle(indices)
	instances = [instances[i] for i in indices]

	dev_instances = instances[:dev_cnt]
	test_instances = instances[dev_cnt:]

	dev_dset = DatasetBERT(filename, args, word2id, device, rel2id = rel2id, instances = dev_instances, verbose=False, instance_only = instance_only, postfix = "cdev")
	test_dset = DatasetBERT(filename, args, word2id, device, rel2id = rel2id, instances = test_instances, verbose=False, instance_only = instance_only, postfix = "test")
	return dev_dset, test_dset



def build_hd_dataset_shuffle(train_dset, cdev_dset,args, word2id, device, rel2id):
	train_hd = DatasetHD(train_dset, cdev_dset, args, word2id, device, rel2id)
	return train_hd


def map_to_ids(tokens, vocab):
		ids = [vocab[t] if t in vocab else UNK_ID for t in tokens]
		return ids

def get_positions(start_idx, end_idx, length):
		""" Get subj/obj relative position sequence. """
		return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
			   list(range(1, length-end_idx))

def sort_all(batch, lens):
	""" Sort all fields by descending order of lens, and return the original indices. """
	unsorted_all = [lens] + [range(len(lens))] + list(batch)
	sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
	return sorted_all[2:], sorted_all[1]

def recover_idx(orig_idx):
	orig2now = [0]*len(orig_idx)
	for idx, orig in enumerate(orig_idx):
		orig2now[orig] = idx
	return orig2now

def eval(pred, labels, weights):
	correct_by_relation = 0
	guessed_by_relation = 0
	gold_by_relation = 0

	# Loop over the data to compute a score
	for idx in range(len(pred)):
		gold = labels[idx]
		guess = pred[idx]

		if gold == NO_RELATION and guess == NO_RELATION:
			pass
		elif gold == NO_RELATION and guess != NO_RELATION:
			guessed_by_relation += weights[gold]
		elif gold != NO_RELATION and guess == NO_RELATION:
			gold_by_relation += weights[gold]
		elif gold != NO_RELATION and guess != NO_RELATION:
			guessed_by_relation += weights[gold]
			gold_by_relation += weights[gold]
			if gold == guess:
				correct_by_relation += weights[gold]

	prec = 0.0
	if guessed_by_relation > 0:
		prec = float(correct_by_relation/guessed_by_relation)
	recall = 0.0
	if gold_by_relation > 0:
		recall = float(correct_by_relation/gold_by_relation)
	f1 = 0.0
	if prec + recall > 0:
		f1 = 2.0 * prec * recall / (prec + recall)

	return prec, recall, f1


def calcEntropy(batch_scores):
	# input: B * L
	# output: B
	batch_probs = nn.functional.softmax(batch_scores)
	return torch.sum(batch_probs * torch.log(batch_probs), dim=1).neg()

def calcInd(batch_probs):
	# input: B * L
	# output: B
	_, ind = torch.max(batch_probs, 1)
	return ind

def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

