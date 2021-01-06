__author__ = 'Maosen'
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel

import utils
from utils import pos2id, ner2id
import sys
from tqdm import tqdm

PRED_MODEL_BILINEAR = 1
PRED_MODEL_CONCATLINEAR = 1

class SBERT(nn.Module):
	def __init__(self, args, rel2id, word_emb=None):
		super(SBERT, self).__init__()
		# arguments
		hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, window_size, num_layers, dropout = \
			args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
			args.position_dim, args.attn_dim, args.window_size, args.num_layers, args.out_drop

		self.pred_model = args.pred_model
		bert_hidden = 768

		self.dual = args.dual
		self.multitask = args.multitask
		# embeddings
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.projection_layer = nn.Linear(bert_hidden, hidden)

		if self.dual or self.multitask:
			if self.dual:
				self.num_output_modules = 4
			elif self.multitask:
				self.num_output_modules = 2

			prednet_list = []
			for mi in range(self.num_output_modules):
				if self.pred_model == PRED_MODEL_BILINEAR:
					pred_model = nn.Bilinear(hidden, hidden, len(rel2id))
				else:
					pred_model = nn.Linear(2*hidden, len(rel2id))

				if mi == 1:
					pred_model.load_state_dict(prednet_list[0].state_dict())
				else:
					pred_model.weight.data.normal_(std=0.001)

				prednet_list.append(pred_model)

			self.prednet_list = nn.ModuleList(prednet_list)
		else:
			if self.pred_model == PRED_MODEL_BILINEAR:
				self.flinear = nn.Bilinear(hidden, hidden, len(rel2id))
			else:
				self.flinear = nn.Linear(2 * hidden, len(rel2id))
			self.flinear.weight.data.normal_(std=0.001)



	def forward(self, inputs):
		words, subj_map, obj_map = inputs

		bert_out, _ = self.bert(words)

		context = self.projection_layer(bert_out)

		vec_subj = torch.bmm(subj_map.unsqueeze(1), context).squeeze(1)
		vec_obj = torch.bmm(obj_map.unsqueeze(1), context).squeeze(1)

		if self.dual or self.multitask:
			out_list_l = []
			for mi in range(self.num_output_modules):
				if self.pred_model == PRED_MODEL_BILINEAR:
					output = self.prednet_list[mi](vec_subj, vec_obj)
				else:
					pred_in = torch.cat((vec_subj, vec_obj))
					output = self.prednet_list[mi](pred_in)
				out_list_l.append(output)
			return out_list_l
		else:
			if self.pred_model == PRED_MODEL_BILINEAR:
				out_logits = self.flinear(vec_subj, vec_obj)
			else:
				pred_in = torch.cat((vec_subj, vec_obj))
				out_logits = self.flinear(pred_in)
			return out_logits





