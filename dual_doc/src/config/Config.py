# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib

from config.BaseConfig import BaseConfig
from utils import Accuracy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from settings import WRITER_DIR
import torch.nn.functional as F


IGNORE_INDEX = -100
WRITER_DIR_DUAL = f"{WRITER_DIR}/docext_dual"


class Config(BaseConfig):
    def __init__(self, args):
        super().__init__(args)
        self.lr = 1e-5

        self.batch_keys += ["context_masks","context_starts", "sent_idxs", "reverse_sent_idxs"]
        self.batch_keys_float += []

    def combine_sents(self, h_idx, t_idx, vertexSet, sents_idx):
        h_t_sent = []
        for ins in vertexSet[h_idx]+vertexSet[t_idx]:
            sent_id = ins['sent_id']
            if sent_id not in h_t_sent:
                h_t_sent.append(sent_id)
        h_t_sent = sorted(h_t_sent)
        #print(sents_idx, h_t_sent)
        combined_sents = []
        for idx in h_t_sent:
            combined_sents += sents_idx[idx]
        combined_sents = combined_sents[:self.combined_sent_limit]
        ret_sent = np.zeros(self.combined_sent_limit) - 1
        ret_sent[:len(combined_sents)] = combined_sents
        #print(ret_sent)
        return ret_sent

    def load_sent_idx(self, ins):
        loaded_sent_idx = ins['sents_idx']
        ret_np = np.zeros((self.sent_limit, self.word_size)) - 1
        reverse_sent_idx = np.zeros((self.max_length)) - 1
        start_idx = 0
        for i, _ in enumerate(loaded_sent_idx):
            _ = _[:self.word_size]
            ret_np[i,:len(_)] = _
            reverse_sent_idx[start_idx:start_idx+len(_)] = list(range(i*self.word_size, i*self.word_size+len(_)))
            start_idx += len(_)
        return ret_np, reverse_sent_idx

    def _get_batch(self, b, data_order, datafile,
                   data_word, data_bert_word, data_pos, data_ner,
                   data_bert_mask, data_bert_starts, ds = False):
        datalen = len(datafile)
        start_id = b * self.batch_size
        cur_bsz = min(self.batch_size, datalen - start_id)
        cur_batch = list(data_order[start_id: start_id + cur_bsz])
        cur_batch.sort(key=lambda x: np.sum(data_word[x] > 0), reverse=True)

        max_length = np.sum(data_word[cur_batch[0]] > 0).item()
        if self.use_bert:
            for x in cur_batch:
                max_length = max(max_length, np.sum(data_bert_word[x] > 0))

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

        #context_char_idxs = np.zeros(shape_txt + (self.char_limit,), dtype=np.int)

        shape_b_ht_l = shape_pair + (max_length,)
        h_mapping = np.zeros(shape_b_ht_l)
        t_mapping = np.zeros(shape_b_ht_l)

        relation_mask = np.zeros(shape_pair)
        relation_label = np.full(shape_pair, IGNORE_INDEX, dtype=np.int)
        relation_multi_label = np.zeros(shape_pair + (self.relation_num,))
        ht_pair_pos = np.zeros(shape_pair, dtype=np.int)

        sent_idxs = np.full((cur_bsz, sent_limit, self.word_size), -1, dtype=np.int)
        reverse_sent_idxs = np.full((cur_bsz, max_length), -1, dtype=np.int)

        max_h_t_cnt = 1
        L_vertex = []
        indexes = []
        vertex_sets = []
        for i, index in enumerate(cur_batch):
            for j in range(max_length):
                if data_word[index, j] == 0:
                    break
                pos_idx[i, j] = j + 1

            if self.use_bert:
                doclen = (data_bert_word[index] > 0).sum()
                context_idxs[i, :doclen] = data_bert_word[index, :doclen]
                context_masks[i, :doclen] = data_bert_mask[index, :doclen]
                context_starts[i, :doclen] = data_bert_starts[index, :doclen]

            else:
                doclen = (data_word[index] > 0).sum()
                context_idxs[i, :doclen] = data_word[index, :doclen]

            context_pos[i, :doclen] = data_pos[index, :doclen]
            context_ner[i, :doclen] = data_ner[index, :doclen]
            #context_char_idxs[i, :doclen] = data_char[index, :doclen]

            ins = datafile[index]
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

            lower_bound = len(ins['na_triple'])
            # random.shuffle(ins['na_triple'])
            # lower_bound = max(20, len(train_tripe)*3)

            lower_bound = min(len(ins['na_triple']), len(train_tripe) * 3)
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
            "distant_supervision":ds,
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
            #'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len],
            'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
            'sent_idxs': sent_idxs[:cur_bsz],
            'sent_lengths': sent_lengths[:cur_bsz],
            'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
        }
        if self.use_bert:
            batch['context_masks'] = context_masks[:cur_bsz, :max_c_len]
            batch['context_starts'] = context_starts[:cur_bsz, :max_c_len]

        return batch


    def _get_train_batch(self, b, train_order):
        return self._get_batch(b, train_order, self.train_file,
                        self.data_train_word, self.data_train_bert_word,
                        self.data_train_pos, self.data_train_ner,
                        self.data_train_bert_mask, self.data_train_bert_starts)

    def get_train_order(self, ds = False):
        if not DEBUG_NOSHUFFLE:
            random.shuffle(self.train_order)
        return self.train_order

    def get_train_batch_old(self):
        random.shuffle(self.train_order)

        context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cuda()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cuda()

        context_masks = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.batch_size, self.max_length).cuda()

        pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()

        context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()


        ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        sent_idxs = torch.LongTensor(self.batch_size, self.sent_limit, self.word_size).cuda()
        reverse_sent_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()

        for b in range(self.train_batches):
            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

            for mapping in [h_mapping, t_mapping]:
                mapping.zero_()

            for mapping in [relation_multi_label, relation_mask, pos_idx]:
                mapping.zero_()

            ht_pair_pos.zero_()

            sent_idxs.zero_()
            sent_idxs -= 1
            reverse_sent_idxs.zero_()
            reverse_sent_idxs -= 1


            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 1


            for i, index in enumerate(cur_batch):
                if self.use_bert:
                    context_idxs[i].copy_(torch.from_numpy(self.data_train_bert_word[index, :]))
                    context_masks[i].copy_(torch.from_numpy(self.data_train_bert_mask[index, :]))
                    context_starts[i].copy_(torch.from_numpy(self.data_train_bert_starts[index, :]))
                else:
                    context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))

                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))


                for j in range(self.max_length):
                    if self.data_train_word[index, j]==0:
                        break
                    pos_idx[i, j] = j+1

                ins = self.train_file[index]
                this_sent_idxs, this_reverse_sent_idxs = self.load_sent_idx(ins)
                sent_idxs[i].copy_(torch.from_numpy(this_sent_idxs))
                reverse_sent_idxs[i].copy_(torch.from_numpy(this_reverse_sent_idxs))
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



                lower_bound = len(ins['na_triple'])
                # random.shuffle(ins['na_triple'])
                # lower_bound = max(20, len(train_tripe)*3)

                lower_bound = min(len(ins['na_triple']), len(train_tripe)*3)
                sel_idx = random.sample(list(range(len(ins['na_triple']))), min(len(ins['na_triple']), lower_bound))
                sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]
                #sel_ins = []
                #for j, (h_idx, t_idx) in enumerate(ins['na_triple'], len(train_tripe)):
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
                #print(max_h_t_cnt)

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)


            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            sent_lengths = (sent_idxs[:cur_bsz] > 0).long().sum(-1)
            #print(reverse_sent_idxs[0])
            #print(sent_idxs, sent_idxs.size())
            #print(sent_lengths, sent_lengths.size())

            batch =  {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths' : input_lengths,
                   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'sent_idxs': sent_idxs[:cur_bsz],
                   'sent_lengths': sent_lengths[:cur_bsz],
                   'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
                   }
            if self.use_bert:
                batch['context_masks']= context_masks[:cur_bsz, :max_c_len].contiguous()
                batch['context_starts']= context_starts[:cur_bsz, :max_c_len].contiguous()

            yield batch

    def get_test_batch(self, contain_relation_multi_label = False):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()
        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()
        sent_idxs = torch.LongTensor(self.test_batch_size, self.sent_limit, self.word_size).cuda()
        reverse_sent_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()

        context_masks = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.test_batch_size, self.max_length).cuda()

        relation_multi_label = torch.FloatTensor(self.test_batch_size, self.h_t_limit, self.relation_num).cuda()

        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()


            ht_pair_pos.zero_()

            sent_idxs.zero_()
            sent_idxs -= 1
            reverse_sent_idxs.zero_()
            reverse_sent_idxs -= 1

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x]>0) , reverse = True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []

            evi_nums = []

            for i, index in enumerate(cur_batch):
                if self.use_bert:
                    context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_word[index, :]))
                    context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))
                    context_starts[i].copy_(torch.from_numpy(self.data_test_bert_starts[index, :]))
                else:
                    context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))

                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]
                this_sent_idxs, this_reverse_sent_idxs = self.load_sent_idx(ins)
                sent_idxs[i].copy_(torch.from_numpy(this_sent_idxs))
                reverse_sent_idxs[i].copy_(torch.from_numpy(this_reverse_sent_idxs))

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

                            #this is for qualitative analysis
                            if contain_relation_multi_label:
                                label = idx2label[(h_idx, t_idx)]
                                for r in label:
                                    relation_multi_label[i, j, r] = 1

                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                evi_num_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in'+self.train_prefix]
                    evi_num_set[(label['h'], label['t'], label['r'])] = len(label['evidence'])

                labels.append(label_set)
                evi_nums.append(evi_num_set)


                L_vertex.append(L)
                indexes.append(index)



            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            sent_lengths = (sent_idxs[:cur_bsz] > 0).long().sum(-1)


            batch =  {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths,
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes,
                   'sent_idxs': sent_idxs[:cur_bsz],
                   'sent_lengths': sent_lengths[:cur_bsz],
                   'reverse_sent_idxs': reverse_sent_idxs[:cur_bsz, :max_c_len],
                   'evi_num_set': evi_nums,
                   }

            if self.use_bert:
                batch['context_masks']= context_masks[:cur_bsz, :max_c_len].contiguous()
                batch['context_starts']= context_starts[:cur_bsz, :max_c_len].contiguous()
            if contain_relation_multi_label:
                batch["relation_multi_label"] = relation_multi_label[:cur_bsz, :max_h_t_cnt]
            yield batch

    def forward(self, model, data):
        context_idxs = data['context_idxs']
        context_pos = data['context_pos']
        h_mapping = data['h_mapping']
        t_mapping = data['t_mapping']
        #relation_label = data['relation_label']
        input_lengths = data['input_lengths']
        #relation_multi_label = data['relation_multi_label']
        relation_mask = data['relation_mask']
        context_ner = data['context_ner']
        #context_char_idxs = data['context_char_idxs']
        ht_pair_pos = data['ht_pair_pos']
        sent_idxs = data['sent_idxs']
        sent_lengths = data['sent_lengths']
        reverse_sent_idxs = data['reverse_sent_idxs']

        if "context_char_idxs" in data:
            context_char_idxs = data["context_char_idxs"]
        else:
            context_char_idxs = None


        dis_h_2_t = ht_pair_pos + 10
        dis_t_2_h = -ht_pair_pos + 10

        if self.use_bert:
            context_masks = data['context_masks']
            context_starts = data['context_starts']
            predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,
                               h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs,
                               sent_lengths, reverse_sent_idxs, context_masks, context_starts)
        else:
            predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)
        return predict_re


    def update_ACC(self, relation_label, output):
        relation_label = relation_label.data.cpu().numpy()

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                label = relation_label[i][j]
                if label < 0:
                    break

                if label == 0:
                    self.acc_NA.add(output[i][j] == label)
                else:
                    self.acc_not_NA.add(output[i][j] == label)

                self.acc_total.add(output[i][j] == label)





    def train(self, model_pattern, model_name):
        ori_model = model_pattern(config = self)
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        ori_model.cuda()
        model = nn.DataParallel(ori_model)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr)
        # nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

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
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.3, 1.0)
        plt.xlim(0.0, 0.4)
        plt.title('Precision-Recall')
        plt.grid(True)

        #model.eval()
        #f1, auc, pr_x, pr_y = self.test(model, model_name)

        for epoch in range(self.max_epoch):
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()

            for data in self.get_train_batch_debug():
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']

                predict_re = self.forward(model, data)
                loss = torch.sum(BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)) /  (self.relation_num * torch.sum(relation_mask))


                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.update_ACC(relation_label, output)

                global_step += 1
                total_loss += loss.item()

                if global_step % self.period == 0 :
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    logging('| epoch {:2d} | step {:5d} |  ms/b {:6.2f} | train loss {:6.4f} | NA acc: {:6.4f} | not NA acc: {:6.4f}  | tot acc: {:4.2f} '.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                    total_loss = 0
                    start_time = time.time()



            if (epoch+1) % self.test_epoch == 0:
                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()
                f1, auc, pr_x, pr_y = self.test(model, model_name, epoch = epoch)
                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)


                if f1 > best_f1:
                    best_f1 = f1
                    best_auc = auc
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)

                    plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join("fig_result", model_name))

        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        print("Finish storing")

    def test(self, model, model_name, epoch = -1, output=False, input_theta=-1, two_phase=False, pretrain_model=None, writer = None):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        predicted_as_zero = 0
        total_ins_num = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')



        for data in self.get_test_batch():
            with torch.no_grad():
                labels = data['labels']
                L_vertex = data['L_vertex']

                titles = data['titles']
                indexes = data['indexes']

                predict_re = self.forward(model, data)

            predict_re = predict_re.data.cpu().numpy()
            #predict_re = predict_re.data.cpu().numpy()
            if two_phase:
                is_rel_exist = is_rel_exist.cpu().numpy()

            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]


                total_recall += len(label)
                for l in label.values():
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
                                        test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )
                                    else:
                                        test_result.append( ((h_idx, t_idx, r) in label, -100.0, intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )
                                else:
                                    test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                            if flag:
                                have_label += 1

                            j += 1

            data_idx += 1


            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

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
        #print(pr_x[f1_pos], pr_y[f1_pos])
        theta = test_result[f1_pos][1]

        if writer is not None:
            writer.add_scalar("test/f1", f1, epoch)
            #writer.add_scalar("train/auc", auc, epoch)

        if input_theta==-1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        if not self.is_test:
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

        if output:
            # output = [x[-4:] for x in test_result[:w+1]]
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
            json.dump(output, open(self.test_prefix + "_index.json", "w"))

        plt.plot(pr_x, pr_y, lw=2, label=model_name)
        plt.legend(loc="upper right")
        if not os.path.exists(self.fig_result_dir):
            os.mkdir(self.fig_result_dir)
        plt.savefig(os.path.join(self.fig_result_dir, model_name))

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train==correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        writer.add_scalar("test/f1", f1)
        writer.add_scalar("train/auc", auc)

        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

        return f1, auc, pr_x, pr_y

    def testall(self, model_pattern, model_name, input_theta, two_phase=False, pretrain_model_name=None):#, ignore_input_theta):
        pretrain_model = None
        if two_phase:
            pretrain_model = model_pattern(config = self)
            pretrain_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, pretrain_model_name)))
            pretrain_model.cuda()
            pretrain_model.eval()

        model = model_pattern(config = self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        model.eval()
        #self.test_anylyse(model, model_name, True, input_theta)
        f1, auc, pr_x, pr_y = self.test(model, model_name, True, input_theta, two_phase, pretrain_model)

    def add_attr(self, attr_list, key, values):
        for i, v in enumerate(values):
            attr_list[key][i].append(v)




