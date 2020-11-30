import pickle
import threading
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sklearn.metrics

from config import Config
from config.BaseConfig import Accuracy, HALF_PRECISION
import matplotlib.pyplot as plt
import random
import itertools

from tensorboardX import SummaryWriter

from config.Config import IGNORE_INDEX
from models import ContextAware, LSTM, BERT_RE
from settings import *
from utils import Accuracy as AccuracyR
from multiprocessing import Manager, Process

from utils import from_cpu_to_cuda_list


def get_expname_hdtune(option_dict):
    model_name = option_dict["model_name"]
    lr = option_dict["learning_rate"]
    tuning_opt = option_dict["tuning_opt"]

    expname = f"hdtune_{model_name}_h{option_dict['hidden_size']}_lr{lr}_to{tuning_opt}"

    if "ner_emb" in option_dict and option_dict["ner_emb"]:
        expname += "_ner"

    if "train_nonerel" in option_dict:
        expname += f"_nr{option_dict['train_nonerel']}"
    if "reduce" in option_dict:
        expname += f"_rd{option_dict['reduce']}"
    if "negativelabel" in option_dict:
        expname += f"_nl{option_dict['negativelabel']}"


    return expname

def get_expname(option_dict):
    if "config" in option_dict and option_dict['config'] == "hdtune":
        return get_expname_hdtune(option_dict)

    model_name = option_dict["model_name"]
    lr = option_dict["learning_rate"]

    expname_common = f"{model_name}_h{option_dict['hidden_size']}_pha{option_dict['proportion_ha']}_lr{lr}"
    if option_dict["num_output_module"] == 4:
        expname = f"{expname_common}_{option_dict['mu_activation']}_wd{option_dict['w_dist']}_sb{option_dict['sanity_bound']}_rm{option_dict['reg_mean']}_dist{option_dict['distance']}_ti{option_dict['twin_init']}"
        if "zeromean_epoch" in option_dict and option_dict["zeromean_epoch"]>0:
            expname += f"_zme{option_dict['zeromean_epoch']}"
    elif option_dict["num_output_module"] == 2:
        expname = f"{expname_common}_om{option_dict['num_output_module']}_dist{option_dict['distance']}"
    else:
        expname = expname_common

    if option_dict["config"] =="dualinterleave":
        expname = "interleave_"+expname


    if "ner_emb" in option_dict and option_dict["ner_emb"]:
        expname += "_ner"
    if "light" in option_dict and option_dict["light"]:
        expname += "_lgt"

    if "train_nonerel" in option_dict:
        expname += f"_nr{option_dict['train_nonerel']}"
    if "reduce" in option_dict:
        expname += f"_rd{option_dict['reduce']}"
    if "negativelabel" in option_dict:
        expname += f"_nl{option_dict['negativelabel']}"



    if "reg_std" in option_dict and option_dict['reg_std']:
        expname += "_rs"
    if "const_mu" in option_dict and option_dict["const_mu"]>0.0:
        expname += f"_cm{option_dict['const_mu']}"

    if "lr_shiftdown" in option_dict and "lr_shiftdown_epoch" in option_dict:
        if option_dict["lr_shiftdown_epoch"] >-1 and option_dict["lr_shiftdown"] < 1.0:
            expname += f"_lsd_e{option_dict['lr_shiftdown_epoch']}_r{option_dict['lr_shiftdown']}"
    if "skip_connection" in option_dict and option_dict["skip_connection"]!=1:
        expname += f"_sc{option_dict['skip_connection']}"

    if "diff_label" in option_dict and option_dict["diff_label"]:
        expname += f"_dl"
        if "sb_prob" in option_dict and option_dict["sb_prob"]>0.0:
            expname += f"_sbp{option_dict['sb_prob']}"
    if "hatrain_partial" in option_dict and option_dict["hatrain_partial"] <1.0:
        expname += f"_trhap{option_dict['hatrain_partial']}"
    if "additional_opt" in option_dict and option_dict["additional_opt"]:
        expname += f"_{option_dict['additional_opt']}"


    return expname



class DualConfig(Config):
    DISTANCE_NO = -1
    DISTANCE_DIFF_GAUSSIAN = 0
    DISTANCE_INFLATION_LOGNORMAL = 1
    DISTANCE_INFLATION_GAUSSIAN = 2
    DISTANCE_POLYNOMIAL_L2 = 3


    LOSS_REDUCTION_SUM = 0
    LOSS_REDUCTION_MEAN = 1

    NEGATIVE_ALL = 0
    NEGATIVE_3TIMES = 1

    name = "dual"

    def __init__(self, args):
        super().__init__(args)
        self.learning_rate = args.learning_rate
        self.mseloss = nn.MSELoss(reduction = "none")
        self.dual_trianing_option = args.dualopt
        self.model_name = args.model_name
        self.num_output_module = args.num_output_module

        print("Dual",self.dual_trianing_option)

        self.acc_NA_ha = Accuracy()
        self.acc_not_NA_ha = Accuracy()
        self.acc_total_ha = Accuracy()

        self.acc_NA_ds = Accuracy()
        self.acc_not_NA_ds = Accuracy()
        self.acc_total_ds = Accuracy()
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        self.w_dist = args.w_dist
        self.diffonins = args.diffonins
        self.diff_logit = args.difflogit
        self.reduce = args.reduce
        self.negativelabel = args.negativelabel
        self.sb_prob = args.sb_prob
        self.diff_label = args.diff_label
        self.distance = args.distance

        self.mu_activation_option = args.mu_activation
        if self.distance == self.DISTANCE_POLYNOMIAL_L2:
            self.mu_activation_f = torch.nn.Softplus()
        elif self.mu_activation_option == "tanh":
            self.mu_activation_f = torch.nn.Tanh()
        elif self.mu_activation_option == "leakyrelu":
            self.mu_activation_f = torch.nn.LeakyReLU()
        elif self.mu_activation_option == "tanhshrink":
            self.mu_activation_f = torch.nn.Tanhshrink()
        elif self.mu_activation_option == "softplus":
            self.mu_activation_f = torch.nn.Softplus()
        elif self.mu_activation_option == "param":
            pass


        if self.distance == DualConfig.DISTANCE_DIFF_GAUSSIAN:
            if self.mu_activation_option == "softplus":
                print("INCOMPATABLE distance-act", self.diff_logit, self.mu_activation_option)
                exit()

        if self.distance in [self.DISTANCE_INFLATION_LOGNORMAL,self.DISTANCE_INFLATION_GAUSSIAN]:
            if self.diff_logit:
                print("INCOMPATABLE diff_logit-distance", self.diff_logit, self.distance)
                exit()

        if self.distance == self.DISTANCE_INFLATION_LOGNORMAL:
            if self.mu_activation_option in []:
                print("INCOMPATABLE distance-act", self.diff_logit, self.mu_activation_option)
                exit()
        self.const_mu = args.const_mu
        self.train_nonerel = args.train_nonerel
        self.use_ner_emb = args.ner_emb
        self.num_cross_encoders = args.num_cross_encoders
        self.skip_connection = args.skip_connection
        self.hatrain_partial = args.hatrain_partial
        self.additional_opt = args.additional_opt

        self.proportion_ha = args.proportion_ha
        self.sanity_bound = args.sanity_bound
        self.batch_size_ha = int(self.proportion_ha * self.batch_size)
        self.batch_size_ds = self.batch_size - self.batch_size_ha
        self._train_order_ds = []

        self.reg_mean = args.reg_mean
        self.reg_std = args.reg_std
        self.cross_encoder = args.cross_encoder

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        self.force_continue = args.force_continue
        self.twin_init = args.twin_init

        self.test_batch_size = self.batch_size

        self.zeromean_epoch = args.zeromean_epoch
        self.hidden_size = args.hidden_size

        self.train_ha = self.proportion_ha > 0.0
        self.train_ds = self.proportion_ha < 1.0

    def mu_activation(self, diff_mean_logit, relation_mask = None):
        if relation_mask is not None and len(relation_mask.shape) == 2:
            relation_mask = relation_mask.unsqueeze(-1)
        if self.mu_activation_option == "param":
            if relation_mask is None:
                mean = diff_mean_logit
            else:
                mean = diff_mean_logit * relation_mask.unsqueeze(-1)
        else:
            mean = self.mu_activation_f(diff_mean_logit)
            if relation_mask is not None:
                mean = mean * relation_mask
            if self.const_mu >0.0:
                mean = mean * self.const_mu
        return mean

    def get_train_batch_debug(self, ha = True, ds = False):
        if ha and ds:
            return itertools.chain(self._get_train_batch_debug(ds = True), self._get_train_batch_debug(ds = False))

        _get_train_batch = self._get_train_batch_debug
        _get_train_batch = self._get_train_batch_mp

        if ha:
            return _get_train_batch(ds = False)
        elif ds:
            return _get_train_batch(ds = True)

    #@conditional_profiler
    def _get_train_batch_debug(self, ds = False):
        train_order = self.get_train_order(ds)
        if ds:
            batch_ids = list(range(self.train_batches_ds))
        else:
            batch_ids = list(range(self.train_batches))

        for b in batch_ids:
            batch = self._get_train_batch(b, train_order, ds = ds)
            if batch['torch']:
                yield batch
            else:
                yield self.batch_from_np2torch(batch)

    def get_test_batch_debug(self, contain_relation_multi_label = False):
        batch_ids = list(range(self.test_batches))

        for b in batch_ids:
            batch = self._get_test_batch(b, contain_relation_multi_label)
            if batch['torch']:
                yield batch
            else:
                yield self.batch_from_np2torch(batch)

    def _get_test_batch(self, b, contain_relation_multi_label):
        data_len = self.test_len
        start_id = b * self.batch_size
        cur_bsz = min(self.batch_size, data_len - start_id)
        #cur_batch = list(self.test_order[start_id: start_id + cur_bsz])
        cur_batch = [(i, np.sum(self.data_test_word[i] > 0).item()) for i in self.test_order[start_id: start_id + cur_bsz]]
        cur_batch.sort(key=lambda x: x[1], reverse=True)
        max_length = cur_batch[0][1]
        sent_limit = self.sent_limit

        if self.use_bert:
            input_lengths = []
            for i, len_w in cur_batch:
                input_lengths.append(max(np.where(self.data_test_bert_word[i]==102)[0].item()+1,len_w))
        else:
            input_lengths = [len_w for i, len_w in cur_batch]
        input_lengths = np.array(input_lengths)
        max_length = max(input_lengths)


        labels = []

        L_vertex = []
        titles = []
        indexes = []

        evi_nums = []

        shape_txt = (cur_bsz, max_length)
        shape_pair = (cur_bsz, self.h_t_limit)


        context_idxs = np.zeros(shape_txt, dtype=np.int)
        context_ner = np.zeros(shape_txt, dtype=np.int)
        context_pos = np.zeros(shape_txt, dtype=np.int)
        pos_idx = np.zeros(shape_txt, dtype=np.int)

        if self.use_bert:
            context_masks = np.zeros(shape_txt, dtype=np.int)
            context_starts = np.zeros(shape_txt, dtype=np.int)


        if HALF_PRECISION:
            float_type = np.float32
        else:
            float_type = np.float

        shape_b_ht_l = shape_pair + (max_length,)
        h_mapping = np.zeros(shape_b_ht_l, dtype = float_type)
        t_mapping = np.zeros(shape_b_ht_l, dtype = float_type)

        relation_mask = np.zeros(shape_pair, dtype = float_type)
        #relation_label = np.full(shape_pair, IGNORE_INDEX, dtype=np.int)
        relation_multi_label = np.zeros(shape_pair + (self.relation_num,), dtype = float_type)
        ht_pair_pos = np.zeros(shape_pair, dtype=np.int)

        sent_idxs = np.full((cur_bsz, self.sent_limit, self.word_size), -1, dtype=np.int)
        reverse_sent_idxs = np.full((cur_bsz, max_length), -1, dtype=np.int)

        relation_multi_label = np.zeros((self.test_batch_size, self.h_t_limit, self.relation_num))
        max_h_t_cnt = 1
        L_vertex = []
        indexes = []
        vertex_sets = []

        data_file = self.test_file
        data_word = self.data_test_word
        data_pos = self.data_test_pos
        data_ner = self.data_test_ner
        if self.use_bert:
            data_bert_word = self.data_test_bert_word
            data_bert_mask = self.data_test_bert_mask
            data_bert_starts = self.data_test_bert_starts

        for i, (index, _) in enumerate(cur_batch):

            for j in range(max_length):
                if data_word[index, j] == 0:
                    break
                pos_idx[i, j] = j + 1

            doclen = input_lengths[i]
            if self.use_bert:
                context_idxs[i, :doclen] = data_bert_word[index, :doclen]
                context_masks[i, :doclen] = data_bert_mask[index, :doclen]
                context_starts[i, :doclen] = data_bert_starts[index, :doclen]
                #cs_idx = context_starts[i, :doclen].nonzero()[0]
                #context_starts_idx[i,:len(cs_idx)] = cs_idx

            else:
                context_idxs[i, :doclen] = data_word[index, :doclen]

            context_pos[i, :doclen] = data_pos[index, :doclen]
            context_ner[i, :doclen] = data_ner[index, :doclen]
            # context_char_idxs[i, :doclen] = data_char[index, :doclen]

            ins = data_file[index]
            this_sent_idxs, this_reverse_sent_idxs = self.load_sent_idx(ins)
            sent_idxs[i, :sent_limit] = this_sent_idxs[:sent_limit]
            reverse_sent_idxs[i, :max_length] = this_reverse_sent_idxs[:max_length]
            # reverse_sent_idxs[i].copy_(torch.from_numpy(this_reverse_sent_idxs))

            idx2label = defaultdict(list)

            for label in ins['labels']:
                idx2label[(label['h'], label['t'])].append(label['r'])

            L = len(ins['vertexSet'])
            titles.append(ins['title'])

            j = 0
            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        hlist = ins['vertexSet'][h_idx]
                        tlist = ins['vertexSet'][t_idx]

                        for h in hlist:
                            h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                        for t in tlist:
                            t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                        relation_mask[i, j] = 1

                        delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                        if delta_dis < 0:
                            ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                        else:
                            ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                        # this is for qualitative analysis
                        if contain_relation_multi_label:
                            label = idx2label[(h_idx, t_idx)]
                            for r in label:
                                relation_multi_label[i, j, r] = 1

                        j += 1

            max_h_t_cnt = max(max_h_t_cnt, j)
            label_set = {}
            evi_num_set = {}
            for label in ins['labels']:
                label_set[(label['h'], label['t'], label['r'])] = label['in' + self.train_prefix]
                evi_num_set[(label['h'], label['t'], label['r'])] = len(label['evidence'])

            labels.append(label_set)
            evi_nums.append(evi_num_set)

            L_vertex.append(L)
            indexes.append(index)

        sent_lengths = (sent_idxs[:cur_bsz] > 0).sum(-1)
        max_c_len = int(input_lengths.max())
        batch = {
            "torch": False,
            "to_cuda": True,
            'context_idxs': context_idxs[:cur_bsz, :max_c_len],
            'context_pos': context_pos[:cur_bsz, :max_c_len],
            'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
            't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
            'labels': labels,
            'L_vertex': L_vertex,
            'input_lengths': input_lengths,
            'pos_idx': pos_idx[:cur_bsz, :max_c_len],
            'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
            'context_ner': context_ner[:cur_bsz, :max_c_len],
             #'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len],
            'titles': titles,
            'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
            'indexes': indexes,
            'sent_idxs': sent_idxs[:cur_bsz],
            'sent_lengths': sent_lengths[:cur_bsz],
            'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
            'evi_num_set': evi_nums,
            'cur_batch': cur_batch
        }
        if self.use_bert:
            batch['context_masks'] = context_masks[:cur_bsz, :max_c_len]
            batch['context_starts'] = context_starts[:cur_bsz, :max_c_len]

        if contain_relation_multi_label:
            batch["relation_multi_label"] = relation_multi_label[:cur_bsz, :max_h_t_cnt]


        return batch



    def get_train_order(self, ds = False):
        if ds:
            train_order = self.train_order_ds
        else:
            train_order = self.train_order

        if not DEBUG_NOSHUFFLE:
            random.shuffle(train_order)

        return train_order

    def get_train_batch_debug(self):
        train_order = self.get_train_order(ds = False)

        for b in range(self.train_batches_ha):
            #t_begin_build_batch_np = datetime.datetime.now()
            batch = self._get_train_batch(b, train_order)
            #t_end_build_batch_np = datetime.datetime.now()
            #print(t_end_build_batch_np- t_begin_build_batch_np)
            if batch['torch']:
                yield batch
            else:
                #yield batch
                yield self.batch_from_np2torch(batch)


    def _get_train_batch(self,b, train_order):
        ha_data_len = self.train_len
        start_id = b * self.batch_size_ha

        if self.batch_size_ha == self.batch_size:
            cur_bsz_ha = min(self.batch_size_ha, ha_data_len - start_id)
            cur_bsz_ds = 0
            cur_bsz = cur_bsz_ha
        else:
            cur_bsz = self.batch_size
            cur_bsz_ha = min(self.batch_size_ha, ha_data_len - start_id)
            cur_bsz_ds = cur_bsz - cur_bsz_ha

        if self.model_name in [LSTM.name, ContextAware.name] or cur_bsz == cur_bsz_ha:
            sort_all = True
        else:
            sort_all = False

        cur_batch_ha = [(i, np.sum(self.data_train_word[i]>0).item(), True) for i in train_order[start_id: start_id + cur_bsz_ha]]
        cur_batch_ds = [(i, np.sum(self.data_train_word_ds[i]>0).item(), False) for i in self.get_train_order_ds(cur_bsz_ds)]


        if sort_all:
            cur_batch = cur_batch_ha + cur_batch_ds
            cur_batch.sort(key = lambda x: x[1], reverse=True)
            max_length = cur_batch[0][1]

        else:
            cur_batch_ha.sort(key=lambda x: x[1], reverse=True)
            cur_batch_ds.sort(key=lambda x: x[1], reverse=True)

            cur_batch = cur_batch_ha + cur_batch_ds
            max_length = max(cur_batch_ha[0][1], cur_batch_ds[0][1])
        is_ha = list(map(lambda x: x[2], cur_batch))

        if self.use_bert:
            sent_lengths = []
            for i, len_w, ha in cur_batch:
                if self.use_bert:
                    if ha:
                        sent_lengths.append(np.where(self.data_train_bert_word[i] == 102)[0][-1].item()+1)
                    else:
                        sent_lengths.append(np.where(self.data_train_bert_word_ds[i] == 102)[0][-1].item() + 1)
                else:
                    sent_lengths.append(len_w)

            max_length = max(sent_lengths)

        sent_limit = self.sent_limit

        shape_txt = (cur_bsz, max_length)
        shape_pair = (cur_bsz, self.h_t_limit)

        context_idxs = np.zeros(shape_txt, dtype=np.int)
        context_ner = np.zeros(shape_txt, dtype=np.int)
        context_pos = np.zeros(shape_txt, dtype=np.int)
        pos_idx = np.zeros(shape_txt, dtype=np.int)

        if self.use_bert:
            context_masks = np.zeros(shape_txt, dtype=np.int)
            context_starts = np.zeros(shape_txt, dtype=np.int)
            #context_starts_idx = np.zeros(shape_txt, dtype=np.int)
            #context_masks_ha = np.zeros(shape_txt, dtype=np.int)
            #context_masks_ds = np.zeros(shape_txt, dtype=np.int)

        # context_char_idxs = np.zeros(shape_txt + (self.char_limit,), dtype=np.int)
        if HALF_PRECISION:
            float_type = np.float32
        else:
            float_type = np.float

        shape_b_ht_l = shape_pair + (max_length,)
        h_mapping = np.zeros(shape_b_ht_l, dtype = float_type)
        t_mapping = np.zeros(shape_b_ht_l, dtype = float_type)

        relation_mask = np.zeros(shape_pair, dtype = float_type)
        relation_label = np.full(shape_pair, IGNORE_INDEX, dtype=np.int)
        relation_multi_label = np.zeros(shape_pair + (self.relation_num,), dtype = float_type)
        ht_pair_pos = np.zeros(shape_pair, dtype=np.int)

        sent_idxs = np.full((cur_bsz, sent_limit, self.word_size), -1, dtype=np.int)
        reverse_sent_idxs = np.full((cur_bsz, max_length), -1, dtype=np.int)

        max_h_t_cnt = 1
        L_vertex = []
        indexes = []
        vertex_sets = []
        for i, (index, _, ha) in enumerate(cur_batch):
            if ha:
                data_file = self.train_file
                data_word = self.data_train_word
                data_pos = self.data_train_pos
                data_ner = self.data_train_ner
                if self.use_bert:
                    data_bert_word = self.data_train_bert_word
                    data_bert_mask = self.data_train_bert_mask
                    data_bert_starts = self.data_train_bert_starts
            else:
                data_file = self.train_file_ds
                data_word = self.data_train_word_ds
                data_pos = self.data_train_pos_ds
                data_ner = self.data_train_ner_ds
                if self.use_bert:
                    data_bert_word = self.data_train_bert_word_ds
                    data_bert_mask = self.data_train_bert_mask_ds
                    data_bert_starts = self.data_train_bert_starts_ds

            for j in range(max_length):
                if data_word[index, j] == 0:
                    break
                pos_idx[i, j] = j + 1

            if self.use_bert:
                doclen = sent_lengths[i]
                context_idxs[i, :doclen] = data_bert_word[index, :doclen]
                context_masks[i, :doclen] = data_bert_mask[index, :doclen]
                context_starts[i, :doclen] = data_bert_starts[index, :doclen]
                #cs_idx = context_starts[i, :doclen].nonzero()[0]
                #context_starts_idx[i,:len(cs_idx)] = cs_idx

            else:
                doclen = (data_word[index] > 0).sum()
                context_idxs[i, :doclen] = data_word[index, :doclen]

            context_pos[i, :doclen] = data_pos[index, :doclen]
            context_ner[i, :doclen] = data_ner[index, :doclen]
            # context_char_idxs[i, :doclen] = data_char[index, :doclen]

            ins = data_file[index]
            this_sent_idxs, this_reverse_sent_idxs = self.load_sent_idx(ins)
            sent_idxs[i, :sent_limit] = this_sent_idxs[:sent_limit]
            reverse_sent_idxs[i, :max_length] = this_reverse_sent_idxs[:max_length]
            # reverse_sent_idxs[i].copy_(torch.from_numpy(this_reverse_sent_idxs))
            labels = ins['labels']
            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])

            train_tripe = list(idx2label.keys())
            for j, (h_idx, t_idx) in enumerate(train_tripe):
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                for h in hlist:
                    h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                for t in tlist:
                    t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                label = idx2label[(h_idx, t_idx)]

                delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                for r in label:
                    relation_multi_label[i, j, r] = 1

                relation_mask[i, j] = 1
                rt = np.random.randint(len(label))
                relation_label[i, j] = label[rt]


            # random.shuffle(ins['na_triple'])
            # lower_bound = max(20, len(train_tripe)*3)

            if self.negativelabel == self.NEGATIVE_ALL:
                lower_bound = len(ins['na_triple'])
                sel_idx = list(range(lower_bound))
            else:
                if self.negativelabel == self.NEGATIVE_3TIMES:
                    lower_bound = max(20, len(train_tripe) * 3)

                lower_bound = min(len(ins['na_triple']), lower_bound)
                sel_idx = random.sample(list(range(len(ins['na_triple']))), min(len(ins['na_triple']), lower_bound))



            sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]
            # sel_ins = []
            # for j, (h_idx, t_idx) in enumerate(ins['na_triple'], len(train_tripe)):
            for j, (h_idx, t_idx) in enumerate(sel_ins, len(train_tripe)):
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                for h in hlist:
                    h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                for t in tlist:
                    t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                relation_multi_label[i, j, 0] = 1
                relation_label[i, j] = 0
                relation_mask[i, j] = 1
                delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
            # print(max_h_t_cnt)

            max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

        input_lengths = (context_idxs[:cur_bsz] > 0).sum(1)
        max_c_len = int(input_lengths.max())

        # print(".")#assert max_c_len == max_length
        max_c_len = max(max_c_len, max_length)

        sent_lengths = (sent_idxs[:cur_bsz] > 0).sum(-1)
        # print(reverse_sent_idxs[0])
        # print(sent_idxs, sent_idxs.size())
        # print(sent_lengths, sent_lengths.size())

        batch = {
            "torch": False,
            "ha_aug": False,
            "to_cuda": False,
            'context_idxs': context_idxs[:cur_bsz, :max_c_len],
            'context_pos': context_pos[:cur_bsz, :max_c_len],
            'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
            't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
            'relation_label': relation_label[:cur_bsz, :max_h_t_cnt],
            'input_lengths': input_lengths,
            'pos_idx': pos_idx[:cur_bsz, :max_c_len],
            'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
            'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
            'context_ner': context_ner[:cur_bsz, :max_c_len],
            # 'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len],
            'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
            'sent_idxs': sent_idxs[:cur_bsz],
            'sent_lengths': sent_lengths[:cur_bsz],
            'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
            'cur_batch':cur_batch,
            'cur_bsz_ha':cur_bsz_ha,
            'cur_bsz_ds':cur_bsz_ds,
            "is_ha":is_ha,
        }
        if self.use_bert:
            batch['context_masks'] = context_masks[:cur_bsz, :max_c_len]
            batch['context_starts'] = context_starts[:cur_bsz, :max_c_len]
            #batch['context_starts_idx'] = context_starts_idx[:cur_bsz, :max_c_len]


        return batch



    def get_optiondict(self, model_pattern):
        if model_pattern is None:
            model_name = self.model_name
        else:
            model_name = model_pattern.name
        option_dict = {
            "config": self.name,
            "train_nonerel":self.train_nonerel,
            "learning_rate": self.learning_rate,
            "model_name":model_name,
            "proportion_ha":self.proportion_ha,
            "distance": self.distance,
            "w_dist": self.w_dist,
            "mu_activation":self.mu_activation_option,
            "sanity_bound": self.sanity_bound,
            "reg_std": self.reg_std,
            "reg_mean": self.reg_mean,
            "distance":self.distance,
            "twin_init":self.twin_init,
            "num_output_module":self.num_output_module,
            "const_mu":self.const_mu,
            "reduce": self.reduce,
            "negativelabel":self.negativelabel,
            "zeromean_epoch":self.zeromean_epoch,
            "hidden_size":self.hidden_size,
            "cross_encoder": self.cross_encoder,
            "num_cross_encoders": self.num_cross_encoders,
            "ner_emb": self.use_ner_emb,
            "additional_opt": self.additional_opt,
            "skip_connection": self.skip_connection,
            "diff_label": self.diff_label,
            "sb_prob": self.sb_prob,
            "hatrain_partial":self.hatrain_partial
            #"light":self.light,
        }
        return option_dict
    def get_expname(self, model_pattern):
        option_dict = self.get_optiondict(model_pattern)
        return get_expname(option_dict)

    def update_ACC(self, relation_label, predict_re_ha_l, predict_re_ds_l, cur_bsz_ha = -1, is_ha = None):
        predict_re_ha = self.sigmoid(predict_re_ha_l.detach())
        output_ha = torch.argmax(predict_re_ha, dim=-1)
        output_ha = output_ha.data.cpu().numpy()

        if predict_re_ds_l is None:
            output_ds = output_ha
        else:
            predict_re_ds = self.sigmoid(predict_re_ds_l.detach())
            output_ds = torch.argmax(predict_re_ds, dim=-1)
            output_ds = output_ds.data.cpu().numpy()

        relation_label = relation_label.data.cpu().numpy()

        for i in range(output_ds.shape[0]):
            if is_ha is None:
                ha = ds = True
                if cur_bsz_ha > 0:
                    ha = i < cur_bsz_ha
                    ds = not ha
            else:
                ha = is_ha[i]
                ds = not ha

            for j in range(output_ds.shape[1]):
                label = relation_label[i][j]
                if label < 0:
                    break

                if ha:
                    if label == 0:
                        self.acc_NA_ha.add(output_ha[i][j] == label)
                    else:
                        self.acc_not_NA_ha.add(output_ha[i][j] == label)
                    self.acc_total_ha.add(output_ha[i][j] == label)

                if ds:
                    if label == 0:
                        self.acc_NA_ds.add(output_ds[i][j] == label)
                    else:
                        self.acc_not_NA_ds.add(output_ds[i][j] == label)
                    self.acc_total_ds.add(output_ds[i][j] == label)

    @conditional_profiler
    def train(self, model_pattern):
        exp_name = self.get_expname(model_pattern)
        if not self.save_name:
            self.save_name = exp_name

        model, optimizer = self.load_model_optimizer(model_pattern, self.pretrain_model)
        # nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        dt = datetime.datetime.now()
        log_dir = f"{dt.day}_{dt.hour}:{dt.minute}_{exp_name}"
        if DEBUG_NOSAVE:
            writer = DummyWriter()
        else:
            writer = SummaryWriter(log_dir=f"{WRITER_DIR_DUAL}/{log_dir}", comment=exp_name)

        self.writer = writer
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        best_auc = 0.0
        best_f1 = 0.0
        best_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", self.save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        # model.eval()
        # f1, auc, pr_x, pr_y = self.test(model, model_name)

        sorted_by_len = self.model_name in ["LSTM", "ContextAware"]
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.acc_NA_ha.clear()
            self.acc_not_NA_ha.clear()
            self.acc_total_ha.clear()

            self.acc_NA_ds.clear()
            self.acc_not_NA_ds.clear()
            self.acc_total_ds.clear()
            loss_epoch = 0.0
            distance_epoch = 0.0
            bc = 0


            # for data in self.get_train_batch_debug():
            for data in self.get_train_batch_debug():
                # print("BC train", bc)
                # distant_supervision = data["distant_supervision"]
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                cur_bsz_ha = data["cur_bsz_ha"]
                is_ha = data["is_ha"]
                # t_begin_forward = datetime.datetime.now()
                output = self.forward(model, data)
                # t_begin_loss = datetime.datetime.now()

                if self.num_output_module == 2:
                    predict_re_ha_l, predict_re_ds_l = output
                    loss, distance = self.get_loss(predict_re_ha_l, predict_re_ds_l,
                                                   relation_label, relation_multi_label, relation_mask, cur_bsz_ha, BCE,
                                                   is_ha, sorted_by_len)
                elif self.num_output_module == 4:
                    predict_re_ha_l, predict_re_ds_l, diff_mean_logit, diff_std_logit = output
                    loss, distance, (mean, std) = self.get_loss(predict_re_ha_l, predict_re_ds_l,
                                                                relation_label, relation_multi_label, relation_mask,
                                                                cur_bsz_ha, BCE,
                                                                is_ha, sorted_by_len,
                                                                zeromean=epoch < self.zeromean_epoch,
                                                                diff_mean_logit=diff_mean_logit,
                                                                diff_std_logit=diff_std_logit)
                else:
                    loss = self.get_loss_single(output, relation_label, relation_multi_label, relation_mask,
                                                cur_bsz_ha, BCE)
                    predict_re_ha_l = output
                    predict_re_ds_l = None
                    distance = 0

                # t_begin_update = datetime.datetime.now()
                loss_epoch += loss.item()
                if isinstance(distance, int) or isinstance(distance, float):
                    distance_epoch += distance
                else:
                    distance_epoch += distance.item()

                self.update_ACC(relation_label, predict_re_ha_l, predict_re_ds_l, cur_bsz_ha, is_ha=is_ha)
                # output = output.data.cpu().numpy()
                # t_begin_backward = datetime.datetime.now()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                '''
                t_end_backward = datetime.datetime.now()
                print("Forward",t_begin_loss - t_begin_forward)
                print("Loss",t_begin_update - t_begin_loss)
                print("Acc",t_begin_backward - t_begin_update)
                print("Backward",t_end_backward - t_begin_backward)
                '''
                global_step += 1
                bc += 1
                total_loss += loss.item()

                if global_step % self.period == 0:
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    log_str = 'epoch {:2d}|step {:5d}|ms/b {:6.2f}|loss {:6.4f}| PaccHA {:6.4f} | NaccHA {:6.4f} | TaccHA {:6.4f}| PaccDS {:6.4f} | NaccDS {:6.4f} | TaccDS {:6.4f}' \
                              ''.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss,
                                        self.acc_not_NA_ha.get(), self.acc_NA_ha.get(), self.acc_total_ha.get(),
                                        self.acc_not_NA_ds.get(), self.acc_NA_ds.get(), self.acc_total_ds.get())
                    logging(log_str)
                    total_loss = 0
                    start_time = time.time()
                    # break

                del loss
                del output
                #break

            writer.add_scalar("train/accN_ha", self.acc_NA_ha.get(), epoch)
            writer.add_scalar("train/accP_ha", self.acc_not_NA_ha.get(), epoch)
            writer.add_scalar("train/accTot_ha", self.acc_total_ha.get(), epoch)
            writer.add_scalar("train/accN_ds", self.acc_NA_ds.get(), epoch)
            writer.add_scalar("train/accP_ds", self.acc_not_NA_ds.get(), epoch)
            writer.add_scalar("train/accTot_ds", self.acc_total_ds.get(), epoch)

            writer.add_scalar("train/loss", loss_epoch / bc, epoch)
            writer.add_scalar("train/distance", distance_epoch / bc, epoch)

            if (epoch + 1) % self.test_epoch == 0 or (epoch + 1) == self.max_epoch:
                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()
                f1, auc, pr_x, pr_y, theta = self.test(model, epoch=epoch, writer=writer)
                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)

                if (epoch > 10 and f1 < best_f1 - 0.1):
                    if not self.force_continue:
                        print("Epoch", epoch)
                        print("F1", f1)
                        break

                if f1 > best_f1:
                    best_f1 = f1
                    best_auc = auc
                    best_epoch = epoch
                    self.save_model(model, optimizer, best=True)

            if (epoch + 1) % self.checkpoint_epoch == 0 or (epoch + 1) == self.max_epoch:
                self.save_model(model, optimizer)

        writer.close()
        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        print("Finish storing")


    def get_loss(self, predict_re_ha_l, predict_re_ds_l, relation_label,
                 relation_multi_label, relation_mask, cur_bsz_ha, BCE, is_ha, sorted_by_len, diff_mean_logit = None, diff_std_logit = None, zeromean = False):

        if predict_re_ha_l.device != relation_multi_label.device:
            relation_multi_label, relation_mask = from_cpu_to_cuda_list([relation_multi_label, relation_mask], predict_re_ha_l.device)

        relation_mask = relation_mask.unsqueeze(2)

        '''
        relation_mask_ha = relation_mask[:cur_bsz_ha]
        relation_mask_ds = relation_mask[cur_bsz_ha:]
        loss_ha = BCE(predict_re_ha_l[:cur_bsz_ha], relation_multi_label[:cur_bsz_ha]) * relation_mask_ha
        loss_ds = BCE(predict_re_ds_l[cur_bsz_ha:], relation_multi_label[cur_bsz_ha:]) * relation_mask_ds
        '''
        if sorted_by_len:
            is_ha = torch.tensor(is_ha, dtype = relation_multi_label.dtype, device = relation_multi_label.device).unsqueeze(-1).unsqueeze(-1)
            loss_ha = BCE(predict_re_ha_l, relation_multi_label) * is_ha
            loss_ds = BCE(predict_re_ds_l, relation_multi_label) * (1.0-is_ha)
            loss = (loss_ha + loss_ds) * relation_mask
        else:
            predict_re_l = torch.cat([predict_re_ha_l[:cur_bsz_ha], predict_re_ds_l[cur_bsz_ha:]])
            loss = BCE(predict_re_l, relation_multi_label)*relation_mask


        if self.train_nonerel:
            rel_begin = 0
        else:
            rel_begin = 1

        #loss_ha = torch.sum(loss_ha[:, :, rel_begin:])
        #loss_ds = torch.sum(loss_ds[:, :, rel_begin:])
        loss = torch.sum(loss[:,:,rel_begin:])
        rel_end = self.relation_num
        relation_num = rel_end - rel_begin

        #loss = loss_ha + loss_ds

        if self.reduce == self.LOSS_REDUCTION_MEAN:
            loss = loss / (relation_num * torch.sum(relation_mask))

        if self.distance == self.DISTANCE_NO:
            distance = 0.0
            return loss, distance
        elif diff_mean_logit is None:
            if self.w_dist > 0.0:
                distance = self.distance_ha_ds(predict_re_ha_l, predict_re_ds_l, relation_mask, relation_multi_label)
                loss = loss + self.w_dist * distance
            else:
                distance = 0.0
            return loss, distance
        else:
            if self.diff_label:
                distance, distribution = self.distance_from_label(predict_re_ha_l, predict_re_ds_l, relation_mask,
                                                diff_mean_logit, diff_std_logit, relation_multi_label, zeromean, cur_bsz_ha, is_ha, sorted_by_len)
            else:
                distance, distribution = self.distance_ha_ds_w_distribution(predict_re_ha_l, predict_re_ds_l, relation_mask,
                                                                        diff_mean_logit, diff_std_logit,
                                                                        relation_multi_label, zeromean)

            loss = loss + self.w_dist * distance
            return loss, distance, distribution

    def get_loss_single(self, predict_re_l, relation_label, relation_multi_label, relation_mask, cur_bsz_ha, BCE):
        if predict_re_l.device != relation_multi_label.device:
            relation_multi_label, relation_mask = from_cpu_to_cuda_list([relation_multi_label, relation_mask], predict_re_l.device)

        relation_mask = relation_mask.unsqueeze(2)
        loss = BCE(predict_re_l, relation_multi_label)*relation_mask

        if self.train_nonerel:
            rel_begin = 0
        else:
            rel_begin = 1
        loss = torch.sum(loss[:, :, rel_begin:])
        rel_end = self.relation_num
        relation_num = rel_end - rel_begin
        if self.reduce == self.LOSS_REDUCTION_MEAN:
            loss = loss / (relation_num * torch.sum(relation_mask))

        return loss

    def distance_ha_ds_w_distribution(self, predict_re_ha_l, predict_re_ds_l, relation_mask, diff_mean_logit, diff_std_logit, relation_multi_label, zero_mean):
        #sanity_bound = 1.0 #Sanity bound
        diff_mean = self.mu_activation(diff_mean_logit,relation_mask= None)
        if zero_mean:
            diff_mean = torch.zeros_like(diff_mean)
        #diff_mean = 0.0
        diff_std = self.softplus(diff_std_logit)+self.sanity_bound

        #distribution = torch.distributions.normal.Normal(diff_mean, diff_std)
        #distance = -distribution.log_prob(predict_re_ds- predict_re_ha)

        #Distance1  diff~Normal(mu,std)

        if self.train_nonerel:
            rel_begin = 0
        else:
            rel_begin = 1
        rel_end = self.relation_num
        relation_num = rel_end - rel_begin


        if self.diff_logit:
            vect_ha = predict_re_ha_l
            vect_ds = predict_re_ds_l
        elif self.distance == self.DISTANCE_INFLATION_LOGNORMAL:
            pass
        else:
            predict_re_ha = self.sigmoid(predict_re_ha_l)
            predict_re_ds = self.sigmoid(predict_re_ds_l)
            vect_ha = predict_re_ha
            vect_ds = predict_re_ds

        if self.distance == self.DISTANCE_DIFF_GAUSSIAN:
            diff = vect_ds - vect_ha
            distance = (diff_mean - diff)/diff_std
            distance = 0.5*(distance ** 2)
            distance = distance + torch.log(diff_std)
        elif self.distance == self.DISTANCE_INFLATION_LOGNORMAL:
            if self.sb_prob >0.0:
                vect_ds = self.sigmoid(predict_re_ds_l)
                vect_ha = self.sigmoid(predict_re_ha_l)
                diff = torch.log(vect_ds + self.sb_prob) - torch.log(vect_ha + self.sb_prob)
            else:
                vect_ds = self.logsigmoid(predict_re_ds_l)
                vect_ha = self.logsigmoid(predict_re_ha_l)
                diff  = vect_ds - vect_ha

            #inflation = vect_ds / (vect_ha + self.sanity_bound)
            #sb = 1e-10
            #diff = torch.log(vect_ds) - torch.log(vect_ha)
            distance = (diff_mean - diff) / diff_std
            distance = 0.5 * (distance ** 2)
            distance = distance + torch.log(diff_std)
            distance = distance + diff
        elif self.distance == self.DISTANCE_POLYNOMIAL_L2:
            exponent = diff_mean
            vect_ha2 = torch.pow(vect_ha, exponent)
            distance = (vect_ha2 - vect_ds)/diff_std
            distance = 0.5 * (distance**2)


        #Distance2
        if self.reg_mean:
            if self.distance == self.DISTANCE_DIFF_GAUSSIAN:#mu~Normal(0,1)
                distance = distance + 0.5*(diff_mean**2)
            elif self.distance == self.DISTANCE_INFLATION_LOGNORMAL:#mu~Normal(1,1)
                distance = distance + 0.5*(diff_mean**2)
            elif self.distance == self.DISTANCE_POLYNOMIAL_L2:#mu~Normal(1,1)
                distance = distance + 0.5*((diff_mean-1.0)**2)

        if self.reg_std:
            distance = distance + 0.5*((diff_std)**2)

        if self.diffonins:
            distance = distance *relation_multi_label
            dat_size = torch.sum(relation_multi_label[:,:,rel_begin:rel_end])
        else:
            distance = distance*relation_mask
            dat_size = relation_num * torch.sum(relation_mask)

        if not self.train_nonerel:
            distance = distance[:,:,rel_begin:rel_end]

        if self.reduce == self.LOSS_REDUCTION_MEAN:
            distance = torch.sum(distance) / dat_size
        elif self.reduce == self.LOSS_REDUCTION_SUM:
            distance = torch.sum(distance)
        else:
            print("ERRRR")

        return distance, (diff_mean, diff_std)


    def load_train_data(self, skip_ds = False):
        if not skip_ds:
            self.train_prefix = "train"
            dt_begin_ds = datetime.datetime.now()
            super().load_train_data()

            self.train_file_ds = self.train_file
            self.data_train_word_ds = self.data_train_word
            self.data_train_pos_ds = self.data_train_pos
            self.data_train_ner_ds = self.data_train_ner

            if self.use_bert:
                self.data_train_bert_word_ds = self.data_train_bert_word
                self.data_train_bert_mask_ds = self.data_train_bert_mask
                self.data_train_bert_starts_ds = self.data_train_bert_starts

            self.train_len_ds = self.train_len
            self.train_order_ds = self.train_order
            self.train_batches_ds = self.train_batches
            dt_end_ds = datetime.datetime.now()
            print("Load DS data", dt_end_ds - dt_begin_ds)
        #'''
        self.train_prefix = "dev_train"
        super().load_train_data()

        if self.hatrain_partial <1.0:
            n_train_rec = int(self.hatrain_partial * self.train_len)
            train_order_tmp = self.train_order[:n_train_rec]
            self.train_order = [train_order_tmp[i%n_train_rec] for i in range(self.train_len)]

        if self.batch_size_ha >0:
            ins_num = len(self.train_file)
            self.train_batches_ha = ins_num//self.batch_size_ha
            if ins_num % self.batch_size_ha:
                self.train_batches_ha += 1

    def get_train_order_ds(self, batchsize):
        if len(self._train_order_ds) < batchsize:
            oldlen = len(self._train_order_ds)
            random.shuffle(self.train_order_ds)
            self._train_order_ds += self.train_order_ds.copy()
            newlen = len(self._train_order_ds)
            print(f"DS new epoch bs{batchsize}:  len {oldlen} -> {newlen}")

        new_order = self._train_order_ds[:batchsize]
        self._train_order_ds = self._train_order_ds[batchsize:]
        return new_order


    def publish(self, model, fname, theta = 0.5):
        print("Publish", fname, theta)
        output_result = []
        for data in self.get_test_batch():
            with torch.no_grad():
                labels = data['labels']
                L_vertex = data['L_vertex']
                relation_mask = data['relation_mask']
                titles = data['titles']
                indexes = data['indexes']

                output = self.forward(model, data)
                if self.num_output_module ==2:
                    predict_re_ha_l, predict_re_ds_l = output
                elif self.num_output_module == 4:
                    predict_re_ha_l, predict_re_ds_l, diff_mean_logit, diff_std_logit = output
                else:
                    predict_re_ha_l = predict_re_ds_l = output

                predict_re_ha = self.sigmoid(predict_re_ha_l)

            predict_re = predict_re_ha.data.cpu().numpy()



            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]
                L = L_vertex[i]
                j = 0

                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            if theta <0.0:
                                rlist = [int(np.argmax(predict_re[i, j]))]
                            else:
                                rlist = ((predict_re[i,j,1:]>=theta).nonzero()[0]+1).tolist()

                            for r in rlist:
                                if r== 0:
                                    continue
                                output_result.append((index, (h_idx, t_idx, r)))
                            j+=1

        output_data = []
        doutput_data_wikidataid = []

        for doc_id, (h, t, r) in output_result:
            title = self.test_file[doc_id]['title']
            rec = {"title": title, "h_idx": h, "t_idx": t, "r": r, "evidence": []}
            output_data.append(rec)

            rec_w = rec.copy()
            rec_w["r"] = self.id2rel[r]
            doutput_data_wikidataid.append(rec_w)


        with open(f"{EXTRACTION_DIR}/{self.test_prefix}_{fname}.json", "w") as f:
             json.dump(output_data,f)

        with open(f"{EXTRACTION_DIR}/{self.test_prefix}_{fname}_w.json", "w") as f:
             json.dump(doutput_data_wikidataid,f)




    def test(self, model, epoch = -1, output=False, input_theta=-1, two_phase=False, pretrain_model=None, writer = None, save = False):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        predicted_as_zero = 0
        total_ins_num = 0

        size_ht_tot =  0


        diff_module = (self.num_output_module == 4)

        if diff_module:
            diff_mean_avg = torch.zeros((self.batch_size, self.h_t_limit, self.relation_num)).cuda()
            diff_std_avg = torch.zeros((self.batch_size, self.h_t_limit, self.relation_num)).cuda()

        #for data in self.get_test_batch():
        for data in self.get_test_batch_debug():
            with torch.no_grad():
                labels = data['labels']
                L_vertex = data['L_vertex']
                relation_mask = data['relation_mask']
                titles = data['titles']
                indexes = data['indexes']

                output_logit = self.forward(model, data)

                if diff_module:
                    predict_re_ha_l, predict_re_ds_l, diff_mean_logit, diff_std_logit = output_logit

                    mean = self.mu_activation(diff_mean_logit,relation_mask)
                    std = (self.softplus(diff_std_logit)+self.sanity_bound)*relation_mask.unsqueeze(-1)
                    mshape = mean.shape
                    diff_mean_avg[:mshape[0], :mshape[1], :mshape[2]] += mean
                    diff_std_avg[:mshape[0], :mshape[1], :mshape[2]] += std
                    size_ht_tot += (relation_mask.sum())
                elif self.num_output_module == 2:
                    predict_re_ha_l, predict_re_ds_l = output_logit
                else:
                    predict_re_ha_l = predict_re_ds_l = output_logit

                predict_re_ha = self.sigmoid(predict_re_ha_l)

            predict_re = predict_re_ha.data.cpu().numpy()
            #predict_re = predict_re.data.cpu().numpy()
            if two_phase:
                is_rel_exist = is_rel_exist.cpu().numpy()

            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]


                total_recall += len(label)
                for (h_idx, t_idx, r) , l in label.items():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                j = 0

                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            r = np.argmax(predict_re[i, j])
                            predicted_as_zero += r==0
                            total_ins_num += 1


                            if (h_idx, t_idx, r) in label:
                                top1_acc += 1

                            flag = False

                            for r in range(1, self.relation_num):
                                intrain = False

                                if (h_idx, t_idx, r) in label:
                                    flag = True
                                    if label[(h_idx, t_idx, r)]==True:
                                        intrain = True
                               # if not intrain:
                                #     test_result_ignore.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                                if two_phase:
                                    if is_rel_exist[i,j,1] > is_rel_exist[i,j,0]:
                                        test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )
                                    else:
                                        test_result.append(((h_idx, t_idx, r) in label, -100.0, intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )
                                else:
                                    test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )


                            if flag:
                                have_label += 1

                            j += 1

            data_idx += 1

            if data_idx % self.period_test == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        if diff_module:
            diff_mean_avg = diff_mean_avg.view((-1, 97)).sum(0) / size_ht_tot

            #iff_std_avg /= self.test_batches
            diff_std_avg = diff_std_avg.view((-1, 97)).sum(0) / size_ht_tot
            list1 = list(range(5))
            list_high = [self.rel2id["P20"], self.rel2id["P36"], self.rel2id["P190"]]
            list_low = [self.rel2id["P35"], self.rel2id["P50"], self.rel2id["P123"]]


        res_single_threshold = self._test(test_result, total_recall, total_recall_ignore, predicted_as_zero, total_ins_num, top1_acc, input_theta, writer,
              epoch, output, save)
        return res_single_threshold


    def _test(self, test_result, total_recall, total_recall_ignore, predicted_as_zero, total_ins_num, top1_acc, input_theta, writer,
              epoch, output, save, relation_wise = False):

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_ and self.save_name:
                with open(os.path.join("log", self.save_name), 'a+') as f_log:
                    f_log.write(s + '\n')

        # test_result_ignore.sort(key=lambda x: x[1], reverse=True)
        test_result.sort(key = lambda x: x[1], reverse=True)

        print ('total_recall', total_recall)
        print('predicted as zero', predicted_as_zero)
        print('total ins num', total_ins_num)
        print('top1_acc', top1_acc)


        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i


        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        print(pr_x[f1_pos], pr_y[f1_pos])
        theta = test_result[f1_pos][1]


        if input_theta==-1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

        if writer is not None:
            writer.add_scalar("test/top1acc",top1_acc, epoch)
            writer.add_scalar("test/precision", pr_x[f1_pos],epoch)
            writer.add_scalar("test/recall", pr_y[f1_pos],epoch)
            writer.add_scalar("test/f1", f1, epoch)
            writer.add_scalar("test/auc", auc, epoch)

        if not self.is_test:
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

        if output:
            # output = [x[-4:] for x in test_result[:w+1]]
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
            json.dump(output, open(self.test_prefix + "_index.json", "w"))
        if save:
            self.save_test_result(None, f1, auc, pr_x, pr_y, theta)

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        correct_ignore = 0
        pos_ignore = 0
        w = 0
        for i, item in enumerate(test_result):
            '''
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train==correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            '''
            if item[2]:
                continue
            if item[0]:
                correct_ignore += 1
            pos_ignore += 1
            p = float(correct_ignore)/pos_ignore
            pr_y.append(p)
            #pr_x.append(float(correct) / total_recall)
            pr_x.append(float(correct_ignore)/ total_recall_ignore)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        if writer is not None:
            writer.add_scalar("test/f1_ign", f1, epoch)
            writer.add_scalar("test/auc_ign", auc, epoch)

        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

        return f1, auc, pr_x, pr_y, theta

    def _test_relationwise(self, test_result, total_recall, total_recall_ignore, total_ins_num, writer,
              epoch, output, save):

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_ and self.save_name:
                with open(os.path.join("log", self.save_name), 'a+') as f_log:
                    f_log.write(s + '\n')

        # test_result_ignore.sort(key=lambda x: x[1], reverse=True)
        #test_result.sort(key = lambda x: x[1], reverse=True)

        n_pos_by_relation, th_by_relation, result_by_relation = self.find_relationwise_theta(test_result, total_recall, total_recall_ignore)

        print ('total_recall', total_recall)
        print('total ins num', total_ins_num)


        correct = 0
        correct_ignore = 0
        n_pos = 0
        n_pos_ignore = 0

        if total_recall == 0:
            total_recall = 1  # for test

        output_result = []
        for r in range(1, self.relation_num):
            th = th_by_relation[r]
            for i, item in enumerate(result_by_relation[r]):
                if th > item[1]:
                    break

                correct += item[0]
                n_pos += 1
                x = item
                output_result.append({'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]})

                if not item[2]:
                    correct_ignore += item[0]
                    n_pos_ignore +=1

        precision = correct/n_pos
        recall = correct/total_recall
        f1 = (2.0 * precision * recall)/ (precision + recall + 1e-20)

        precision_ignore = correct_ignore/n_pos_ignore
        recall_ignore = correct_ignore/total_recall_ignore
        f1_ignore = (2.0 * precision_ignore * recall_ignore)/ (precision_ignore + recall_ignore + 1e-20)


        if writer is not None:
            writer.add_scalar("test_rwt/precision", precision, epoch)
            writer.add_scalar("test_rwt/recall", recall, epoch)
            writer.add_scalar("test_rwt/f1", f1, epoch)
            writer.add_scalar("test_rwt/precision_ign", precision_ignore, epoch)
            writer.add_scalar("test_rwt/recall_ign", recall_ignore, epoch)
            writer.add_scalar("test_rwt/f1_ignore", f1_ignore, epoch)


        if output:
            json.dump(output_result, open(self.test_prefix + "_index_rwt.json", "w"))
        if save:
            pass
            #self.save_test_result(None, f1, auc, pr_x, pr_y, theta, accuracy_by_rel)


        logging('F1 {:3.4f} | F1_ign {:3.4f} |P {:3.4f} | P_ign {:3.4f} |R {:3.4f} | R_ign {:3.4f} '.format(f1, f1_ignore, precision, precision_ignore, recall, recall_ignore))
        return th_by_relation, f1, f1_ignore, precision, precision_ignore, recall, recall_ignore

    def save_test_result(self, model_pattern, f1, auc, pr_x, pr_y, theta):
        if not os.path.exists(TEST_RESULT_DIR):
            os.mkdir(TEST_RESULT_DIR)

        result = {"f1": f1,
                  "auc": auc,
                  "theta": theta}
        pickle.dump(result,open(f"{TEST_RESULT_DIR}/{self.get_expname(model_pattern)}.pkl","wb"))

        np.save(f"{TEST_RESULT_DIR}/{self.get_expname(model_pattern)}_pr_x.npy",pr_x)
        np.save(f"{TEST_RESULT_DIR}/{self.get_expname(model_pattern)}_pr_y.npy",pr_y)




class DummyWriter():
    def __init__(self, *args, **kwags): print("Caution: nothing will be recorded")

    def nop(*args, **kwags): pass

    def __getattr__(self, _): return self.nop

if __name__ == "__main__":
    from train import get_ext_parser
    parser = get_ext_parser()
    args = parser.parse_args()
    config = DualConfigShuffle(args)

    config.save_test_result(None,0.5,0.5,[1.0,0.5,0.0],[0.0,0.5,1.0],0.5,[])
