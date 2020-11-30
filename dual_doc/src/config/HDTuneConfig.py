import datetime
import pickle
import random
import time
from collections import defaultdict

import sklearn
import torch
from tensorboardX import SummaryWriter
from torch import nn
import numpy as np
from torch.nn import DataParallel

from config import Config
from config.Config import IGNORE_INDEX
from config.BaseConfig import Accuracy
from config.DualConfig import DummyWriter, DualConfig, get_expname
from settings import *
from utils import from_cpu_to_cuda_list

#TUNING_OPTION_MAX_TH = 1
#TUNING_OPTION_ENTROPY_TH = 2
TUNING_OPTION_BIAS_SET = 3
TUNING_OPTION_BIAS_FIX = 4







class HDTuneConfig(DualConfig):
    name = "hdtune"
    def __init__(self, args):
        super().__init__(args)
        self.learning_rate = args.learning_rate
        self.model_name = args.model_name
        self.tuning_opt = args.tuning_opt
        print("HDTuneConfig", self.tuning_opt)
        self._train_order_ds = []


        self.acc_NA_ha = Accuracy()
        self.acc_not_NA_ha = Accuracy()
        self.acc_total_ha = Accuracy()

        self.acc_NA_ds = Accuracy()
        self.acc_not_NA_ds = Accuracy()
        self.acc_total_ds = Accuracy()

        self.reduce = args.reduce
        self.negativelabel = args.negativelabel
        self.hidden_size = args.hidden_size
        self.cross_encoder = args.cross_encoder
        self.checkpoint_epoch = 5

    def get_expname(self, model_pattern):
        if model_pattern is None:
            model_name = self.model_name
        else:
            model_name = model_pattern.name
        option_dict = {
            "config": self.name,
            "train_nonerel": self.train_nonerel,
            "learning_rate": self.learning_rate,
            "model_name": model_name,
            "negativelabel": self.negativelabel,
            "hidden_size": self.hidden_size,
            "cross_encoder": self.cross_encoder,
            "tuning_opt": self.tuning_opt,
            "reduce": self.reduce,
            "ner_emb": self.use_ner_emb
        }
        return get_expname(option_dict)

    def train(self, model_pattern):
        model, optimizer = self.load_model_optimizer(model_pattern, self.pretrain_model)
        self.compute_bias()
        self.tuning_before_training(model)

        BCE = nn.BCEWithLogitsLoss(reduction='none')
        exp_name = self.get_expname(model_pattern)
        if not self.save_name:
            self.save_name = exp_name

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



        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.acc_NA_ha.clear()
            self.acc_not_NA_ha.clear()
            self.acc_total_ha.clear()

            self.acc_NA_ds.clear()
            self.acc_not_NA_ds.clear()
            self.acc_total_ds.clear()
            loss_epoch = 0.0
            bc = 0
            for data in self.get_train_batch_ds_only():
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                is_ha = data['is_ha']
                # t_begin_forward = datetime.datetime.now()
                predict_re_l = self.forward(model, data)

                loss = self.get_loss(predict_re_l, relation_label, relation_multi_label, relation_mask, BCE)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                total_loss += loss.item()

                self.update_ACC(relation_label,predict_re_l, predict_re_l, 0, is_ha)
                bc += 1
                loss_epoch += loss.item()
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
                    writer.add_scalar("train/loss", cur_loss, global_step)
                    #break


            writer.add_scalar("train/accN_ds", self.acc_NA_ds.get(), epoch)
            writer.add_scalar("train/accP_ds", self.acc_not_NA_ds.get(), epoch)
            writer.add_scalar("train/accTot_ds", self.acc_total_ds.get(), epoch)
            writer.add_scalar("train/loss_e", loss_epoch / bc, epoch)

            f1, auc, pr_x, pr_y, theta = self.test(model, epoch, writer)
            if f1 >best_f1:
                best_f1 = f1
                best_epoch = epoch
                self.save_model(model, optimizer, best = True)

            if (epoch+1)% self.checkpoint_epoch== 0 or (epoch+1) == self.max_epoch:
                self.save_model(model, optimizer)

        writer.close()


    def test(self, model, epoch =-1, writer = None, original = False, save = False):
        self.compute_bias()
        if original:
            res = self.test_single(model, epoch, writer=writer)
        else:
            self.tuning_before_test(model)
            res = self.test_single(model, epoch, writer=writer)
            self.tuning_recover_after_test(model)

        if not os.path.exists(TEST_RESULT_DIR):
            os.mkdir(TEST_RESULT_DIR)
        f1, auc, pr_x, pr_y, theta = res
        result = {"f1": f1,
                  "auc": auc,
                  "theta": theta}
        if save:
            res_fname = self.get_expname(None)
            if original:
                res_fname += "_original"
            pickle.dump(result,open(f"{TEST_RESULT_DIR}/{res_fname}.pkl","wb"))

            np.save(f"{TEST_RESULT_DIR}/{res_fname}_pr_x.npy",pr_x)
            np.save(f"{TEST_RESULT_DIR}/{res_fname}_pr_y.npy",pr_y)

        return res

    def publish(self, model, fname, theta = 0.5, original = False):
        if original:
            res = super().publish(model, fname + "_original", theta)
        else:
            self.tuning_before_test(model)
            res = super().publish(model, fname, theta)
            self.tuning_recover_after_test(model)
        return res




    def test_single(self, model, epoch = -1, output=False, input_theta=-1, writer = None, save = False):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        total_ins_num = 0


        #for data in self.get_test_batch():
        for data in self.get_test_batch_debug():
            with torch.no_grad():
                labels = data['labels']
                L_vertex = data['L_vertex']
                relation_mask = data['relation_mask']
                titles = data['titles']
                indexes = data['indexes']

                output_logit = self.forward(model, data)
                predict_re = self.sigmoid(output_logit)

            predict_re = predict_re.data.cpu().numpy()


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
                            for r in range(1, self.relation_num):
                                intrain = False

                                if (h_idx, t_idx, r) in label:
                                    flag = True
                                    if label[(h_idx, t_idx, r)]:
                                        intrain = True
                                test_result.append(((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                            j += 1

            data_idx += 1

            if data_idx % self.period_test == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()


        f1, auc, pr_x, pr_y, theta = self._test(test_result, total_recall, total_recall_ignore, total_ins_num, top1_acc, input_theta, writer,
              epoch, output, save)
        return f1, auc, pr_x, pr_y, theta


    def _test(self, test_result, total_recall, total_recall_ignore, total_ins_num, top1_acc, input_theta, writer,
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


    def compute_pr(self, datfile):
        pr = np.zeros(self.relation_num)

        pair_count = 0
        for doc in datfile:
            n_vertex = len(doc['vertexSet'])
            pair_count += n_vertex*(n_vertex-1)

            for label in doc['labels']:
                pr[label['r']] += 1
        pr/=pair_count
        pr[0] = 1e-10
        return pr

    def compute_bias(self):
        if not hasattr(self, "prh"):
            self.prh = self.compute_pr(self.train_file)
            self.prd = self.compute_pr(self.train_file_ds)

    def tuning_before_training(self, model):
        if isinstance(model, DataParallel):
            model = model.module
        if self.tuning_opt == TUNING_OPTION_BIAS_FIX:
            bias = np.log(self.prd)
            model.fix_prediction_bias(bias)

    def tuning_before_test(self, model):
        if isinstance(model, DataParallel):
            model = model.module
        if self.tuning_opt == TUNING_OPTION_BIAS_FIX:
            bias = np.log(self.prh)
            model.fix_prediction_bias(bias)
        elif self.tuning_opt == TUNING_OPTION_BIAS_SET:
            delta_bias = np.log(self.prh) - np.log(self.prd)
            model.add_prediction_bias(delta_bias)

    def tuning_recover_after_test(self, model):
        if isinstance(model, DataParallel):
            model = model.module
        if self.tuning_opt == TUNING_OPTION_BIAS_FIX:
            bias = np.log(self.prd)
            model.fix_prediction_bias(bias)
        elif self.tuning_opt == TUNING_OPTION_BIAS_SET:
            delta_bias = np.log(self.prh) - np.log(self.prd)
            model.add_prediction_bias(-delta_bias)



    def get_loss(self, predict_re_l, relation_label, relation_multi_label, relation_mask, BCE):
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

    def get_train_batch_ds_only(self):
        data_len = self.train_len_ds

        num_batches = self.train_len_ds//self.batch_size
        if self.train_len_ds%self.batch_size != 0:
            num_batches += 1
        for b in range(num_batches):
            start_id =  b * self.batch_size
            cur_bsz = min(self.batch_size, data_len - start_id)
            cur_batch = [(i, np.sum(self.data_train_word_ds[i] > 0).item(), False) for i in
                            self.get_train_order_ds(cur_bsz)]

            cur_batch.sort(key=lambda x: x[1], reverse=True)
            max_length = cur_batch[0][1]


            is_ha = list(map(lambda x: x[2], cur_batch))

            if self.use_bert:
                sent_lengths = []
                for i, len_w, ha in cur_batch:
                    if self.use_bert:
                        if ha:
                            sent_lengths.append(np.where(self.data_train_bert_word[i]==102)[0].item()+1)
                        else:
                            sent_lengths.append(np.where(self.data_train_bert_word_ds[i] == 102)[0].item() + 1)
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

            batch = {
                "torch": False,
                "ha_aug": False,
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
                'cur_bsz_ha':0,
                'cur_bsz_ds':cur_batch,
                "is_ha":is_ha,
            }
            if self.use_bert:
                batch['context_masks'] = context_masks[:cur_bsz, :max_c_len]
                batch['context_starts'] = context_starts[:cur_bsz, :max_c_len]
                #batch['context_starts_idx'] = context_starts_idx[:cur_bsz, :max_c_len]

            batch = self.batch_from_np2torch(batch)
            yield batch

