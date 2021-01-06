'''
Model wrapper for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler

import utils
from models.palstm import PositionAwareRNN
from models.bgru import BGRU
from models.cnn import CNN
from models.pcnn import PCNN
from models.lstm import LSTM
import json

from models.sbert import SBERT


class Model(object):
	def __init__(self, args, device, rel2id, word_emb=None):
		lr = args.lr
		lr_decay = args.lr_decay
		weight_decay = args.weight_decay
		self.cpu = torch.device('cpu')
		self.device = device
		self.args = args
		self.rel2id = rel2id
		self.max_grad_norm = args.max_grad_norm
		self.dual = args.dual
		self.multitask = args.multitask
		self.union = args.union

		self.softmax = nn.Softmax(dim = 1)
		self.logsoftmax = nn.LogSoftmax(dim = 1)
		self.tanh = nn.Tanh()
		self.tanhshrink = nn.Tanhshrink()
		self.softplus = nn.Softplus()
		if args.model == 'pa_lstm':
			self.model = PositionAwareRNN(args, rel2id, word_emb)
		elif args.model == 'bgru':
			self.model = BGRU(args, rel2id, word_emb)
		elif args.model == 'cnn':
			self.model = CNN(args, rel2id, word_emb)
		elif args.model == 'pcnn':
			self.model = PCNN(args, rel2id, word_emb)
		elif args.model == 'lstm':
			self.model = LSTM(args, rel2id, word_emb)
		elif args.model == "sbert":
			self.model = SBERT(args, rel2id)
		else:
			raise ValueError
		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss()
		self.criterion_noreduce = nn.CrossEntropyLoss(reduction = "none")

		if args.fix_bias:
			if args.dual:
				raise ValueError("dual & fix bias")
			self.model.flinear.bias.requires_grad = False
		if args.dual:
			self.w_dist = args.w_dist
			self.use_dev = args.use_dev

		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		# self.parameters = self.model.parameters()
		#self.optimizer = torch.optim.Adam(self.parameters, 1e-3)
		if args.optimizer == "SGD":
			self.optimizer = torch.optim.SGD(self.parameters, lr)
		elif args.optimizer == "Adam":
			self.optimizer = torch.optim.Adam(self.parameters, lr, weight_decay=weight_decay)
			
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=lr_decay)
		self.diffall = args.diffall if "diffall" in args else False

	def update_lr(self, valid_loss):
		if isinstance(self.optimizer, torch.optim.Adam):
			print("Adam")
		else:
			self.scheduler.step(valid_loss)
			print("LR", self.optimizer.param_groups[0]['lr'])

	def update(self, batch, penalty=False, weight=1.0, col_label = 5, col_isha = 7):
		inputs = [p.to(self.device) for p in batch[:col_label]]
		labels = batch[col_label].to(self.device)
		self.model.train()
		logits = self.model(inputs)
		if self.dual:
			is_ha = batch[col_isha]
			loss = self.get_loss_dual(logits, labels, is_ha)
		elif self.multitask:
			is_ha = batch[col_isha]
			loss = self.get_loss_dual(logits, labels, is_ha, add_distance = False)
		else:
			loss = self.get_loss(logits, labels)
		# batch_ent = utils.calcEntropy(logits)
		# ent = torch.sum(batch_ent) / len(batch_ent)
		# if penalty:
		# 	loss = loss - ent*weight
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
		self.optimizer.step()
		return loss.item()

	def get_loss(self, logits, labels):
		if self.dual:
			print("TODO: Implement")
		else:
			loss = self.criterion(logits, labels)
		return loss

	def compute_distance(self, pred_logit_ha, pred_logit_ds, mu_logit, sigma_logit, labels):
		diff_mu = self.tanh(mu_logit)
		diff_sigma = self.softplus(sigma_logit) + 1e-5

		pred_ha = self.logsoftmax(pred_logit_ha)
		pred_ds = self.logsoftmax(pred_logit_ds)

		diff = pred_ds - pred_ha
		distance = (diff_mu - diff) /diff_sigma
		distance = 0.5 * (distance **2)
		distance = distance + torch.log(diff_sigma)
		distance = distance + diff

		distance = distance + 0.5 * (diff_mu**2)
		distance = distance + 0.5 * (diff_sigma**2)
		if not self.diffall:
			ids = torch.arange(labels.shape[0])
			distance = distance[ids,labels]
		return distance.mean()



	def get_loss_dual(self, logits, labels, is_ha, add_distance = True):
		pred_logit_ha, pred_logit_ds = logits[:2]

		if isinstance(is_ha, torch.Tensor):
			is_ha = is_ha.to(self.device)
			loss_ha = self.criterion_noreduce(pred_logit_ha, labels) * is_ha
			loss_ds = self.criterion_noreduce(pred_logit_ds, labels) * (1.0 - is_ha)
			loss = (loss_ha+loss_ds).mean()
		elif is_ha:
			loss = self.criterion(pred_logit_ha, labels)
		else:
			loss = self.criterion(pred_logit_ds, labels)

		if add_distance:
			mu_logit, sigma_logit = logits[2:]
			distance = self.compute_distance(pred_logit_ha, pred_logit_ds, mu_logit, sigma_logit, labels)

			loss += self.w_dist * distance

		return loss

	def get_bias(self):
		return self.model.flinear.bias.data

	def set_bias(self, bias):
		if self.dual:
			raise  ValueError
		if bias.device != self.device:
			bias = bias.to(self.device)
		self.model.flinear.bias.data = bias

	def predict(self, batch, col_label=5, col_origidx=6, ds = False):
		inputs = [p.to(self.device) for p in batch[:col_label]]
		labels = batch[col_label].to(self.cpu)
		orig_idx = batch[col_origidx]
		if self.dual or self.multitask:
			out_i = 1 if ds else 0
			logits = self.model(inputs)[out_i].to(self.cpu)
		else:
			logits = self.model(inputs).to(self.cpu)
		loss = self.criterion(logits, labels)
		pred = torch.argmax(logits, dim=1).to(self.cpu)
		# corrects = torch.eq(pred, labels)
		# acc_cnt = torch.sum(corrects, dim=-1)
		recover_idx = utils.recover_idx(orig_idx)
		logits = [logits[idx].tolist() for idx in recover_idx]
		pred = [pred[idx].item() for idx in recover_idx]
		labels = [labels[idx].item() for idx in recover_idx]
		return logits, pred, labels, loss.item()

	def eval(self, dset, vocab=None, col_label=5, col_origidx = 6, output_false_file=None, output_label_file=None, weights=None, ds = False):
		if weights is None:
			weights = [1.0] * len(dset.rel2id)

		rel_labels = ['']*len(dset.rel2id)
		for label, id in dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		loss = 0.0

		for idx, batch in enumerate(dset.batched_data):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch, col_label, col_origidx, ds = ds)
			pred += pred_b
			labels += labels_b
			loss += loss_b

			if output_false_file is not None and vocab is not None:
				if len(batch) == 7:
					all_words, pos, ner, subj_pos, obj_pos, labels_, _ = batch
				else:
					all_words, subj_map, obj_map, labels_, _ = batch
					continue

				all_words = all_words.tolist()
				output_false_file.write('\n')
				for i, word_ids in enumerate(all_words):
					if labels[i] != pred[i]:
						length = 0
						for wid in word_ids:
							if wid != utils.PAD_ID:
								length += 1
						words = [vocab[wid] for wid in word_ids[:length]]
						sentence = ' '.join(words)

						subj_words = []
						for sidx in range(length):
							if subj_pos[i][sidx] == 0:
								subj_words.append(words[sidx])
						subj = '_'.join(subj_words)

						obj_words = []
						for oidx in range(length):
							if obj_pos[i][oidx] == 0:
								obj_words.append(words[oidx])
						obj = '_'.join(obj_words)

						output_false_file.write('%s\t%s\t%s\t%s\t%s\n' % (sentence, subj, obj, rel_labels[pred[i]], rel_labels[labels[i]]))

		if output_label_file is not None and vocab is not None:
			output_label_file.write(json.dumps(pred) + '\n')
			output_label_file.write(json.dumps(labels) + '\n')


		loss /= len(dset.batched_data)
		return loss, utils.eval(pred, labels, weights)

	def TuneEntropyThres(self, test_dset, noneInd=utils.NO_RELATION, ratio=0.2, cvnum=100, col_label=5, col_origidx = 6,):
		'''
		Tune threshold on test set
		'''
		rel_labels = [''] * len(test_dset.rel2id)
		for label, id in test_dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		scores = []
		loss = 0.0
		for idx, batch in enumerate(test_dset.batched_data):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch, col_label = col_label, col_origidx= col_origidx)
			pred += pred_b
			labels += labels_b
			scores += scores_b
			loss += loss_b
		loss /= len(test_dset.batched_data)

		# start tuning
		scores = torch.tensor(scores)
		f1score = 0.0
		recall = 0.0
		precision = 0.0

		pre_ind = utils.calcInd(scores)
		pre_entropy = utils.calcEntropy(scores)
		valSize = int(np.floor(ratio * len(pre_ind)))
		data = [[pre_ind[ind], pre_entropy[ind], labels[ind]] for ind in range(0, len(pre_ind))]

		for cvind in tqdm(range(cvnum)):
			random.shuffle(data)
			val = data[0:valSize]
			eva = data[valSize:]

			# find best threshold
			max_ent = max(val, key=lambda t: t[1])[1]
			min_ent = min(val, key=lambda t: t[1])[1]
			stepSize = (max_ent - min_ent) / 100
			thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
			ofInterest = 0
			for ins in val:
				if ins[2] != noneInd:
					ofInterest += 1
			bestThreshold = float('nan')
			bestF1 = float('-inf')
			for threshold in thresholdList:
				corrected = 0
				predicted = 0
				for ins in val:
					if ins[1] < threshold and ins[0] != noneInd:
						predicted += 1
						if ins[0] == ins[2]:
							corrected += 1
				curF1 = 2.0 * corrected / (ofInterest + predicted)
				if curF1 > bestF1:
					bestF1 = curF1
					bestThreshold = threshold
			ofInterest = 0
			corrected = 0
			predicted = 0
			for ins in eva:
				if ins[2] != noneInd:
					ofInterest += 1
				if ins[1] < bestThreshold and ins[0] != noneInd:
					predicted += 1
					if ins[0] == ins[2]:
						corrected += 1

			f1score += (2.0 * corrected / (ofInterest + predicted))
			recall += (1.0 * corrected / ofInterest)
			precision += (1.0 * corrected / (predicted + 0.00001))


		f1score /= cvnum
		recall /= cvnum
		precision /= cvnum

		return loss, f1score, recall, precision

	def TuneMaxThres(self, test_dset, noneInd=utils.NO_RELATION, ratio=0.2, cvnum=100, col_label=5, col_origidx = 6,):
		'''
		Tune threshold on test set
		'''
		rel_labels = [''] * len(test_dset.rel2id)
		for label, id in test_dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		scores = []
		loss = 0.0
		for idx, batch in enumerate(test_dset.batched_data):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch, col_label = col_label, col_origidx = col_origidx)
			pred += pred_b
			labels += labels_b
			scores += scores_b
			loss += loss_b
		loss /= len(test_dset.batched_data)

		# start tuning
		scores = torch.tensor(scores)
		f1score = 0.0
		recall = 0.0
		precision = 0.0

		pre_prob, pre_ind = torch.max(scores, 1)
		valSize = int(np.floor(ratio * len(pre_ind)))
		data = [[pre_ind[ind], pre_prob[ind], labels[ind]] for ind in range(0, len(pre_ind))]
		for cvind in tqdm(range(cvnum)):
			random.shuffle(data)
			val = data[0:valSize]
			eva = data[valSize:]

			# find best threshold
			max_ent = max(val, key=lambda t: t[1])[1]
			min_ent = min(val, key=lambda t: t[1])[1]
			stepSize = (max_ent - min_ent) / 100
			thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
			ofInterest = 0
			for ins in val:
				if ins[2] != noneInd:
					ofInterest += 1
			bestThreshold = float('nan')
			bestF1 = float('-inf')
			for threshold in thresholdList:
				corrected = 0
				predicted = 0
				for ins in val:
					if ins[1] > threshold and ins[0] != noneInd:
						predicted += 1
						if ins[0] == ins[2]:
							corrected += 1
				curF1 = 2.0 * corrected / (ofInterest + predicted)
				if curF1 > bestF1:
					bestF1 = curF1
					bestThreshold = threshold

			ofInterest = 0
			corrected = 0
			predicted = 0
			for ins in eva:
				if ins[2] != noneInd:
					ofInterest += 1
				if ins[1] > bestThreshold and ins[0] != noneInd:
					predicted += 1
					if ins[0] == ins[2]:
						corrected += 1
			f1score += (2.0 * corrected / (ofInterest + predicted))
			recall += (1.0 * corrected / ofInterest)
			precision += (1.0 * corrected / (predicted + 0.00001))

		f1score /= cvnum
		recall /= cvnum
		precision /= cvnum

		return loss, f1score, recall, precision

	def save(self, filename, epoch):
		params = {
			'model': self.model.state_dict(),
			'config': self.args,
			'epoch': epoch
		}
		try:
			torch.save(params, filename)
			print("Epoch {}, model saved to {}".format(epoch, filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")
		# json.dump(vars(self.args), open('%s.json' % filename, 'w'))

	def count_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def load(self, filename):
		params = torch.load(filename)
		if type(params).__name__ == 'dict' and 'model' in params:
			self.model.load_state_dict(params['model'])
		else:
			self.model.load_state_dict(params)






