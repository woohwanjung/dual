import json
import os
import pickle

import numpy as np
import torch
import datetime
from multiprocessing import Manager, Process

from settings import conditional_profiler, HALF_PRECISION

from utils import load_json

try:
    from pytorch_transformers import BertTokenizer
except:
    BertTokenizer = None
from settings import EXT_DIR
from torch import optim, nn
import multiprocessing as mp
import threading



class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0






class BaseConfig():
    def __init__(self, args):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = 'prepro_data'

        self.use_gpu = True
        self.is_training = True
        self.max_length = 512 #+160
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length
        self.relation_num = 97

        self.use_bag = False

        self.coref_size = 20
        self.entity_type_size = 20
        self.max_epoch = 20
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = f"{EXT_DIR}/checkpoints"
        self.fig_result_dir = './fig_result'
        self.test_epoch = 5
        self.checkpoint_epoch = 50
        self.pretrain_model = None

        self.word_size = 100
        self.epoch_range = None
        self.cnn_drop_prob = 0.5  # for cnn
        self.keep_prob = 0.8  # for lstm

        self.period = 50
        self.period_test = 50

        self.batch_size = args.batch_size
        # self.test_batch_size = 40
        self.h_t_limit = 1800

        self.test_batch_size = self.batch_size
        self.test_relation_limit = 1800
        self.char_limit = 16
        self.sent_limit = 25
        # self.combined_sent_limit = 200
        self.dis2idx = np.zeros((self.max_length), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.use_bert = args.use_bert
        self.train_bert = True
        if hasattr(args,"train_bert"):
            self.train_bert = args.train_bert
        print("Train bert", self.train_bert)
        if not os.path.exists("log"):
            os.mkdir("log")

        self.batch_keys = ['context_idxs', 'context_pos', 'context_ner',
                           'relation_label', 'ht_pair_pos', 'pos_idx',
                           'input_lengths', 'context_char_idxs']
        self.batch_keys_float = ['h_mapping', 't_mapping', 'relation_multi_label', 'relation_mask']
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_word2id = {word:wid for wid, word in self.tokenizer.vocab.items()}
            self.bert_id2word= {wid: word for wid, word in self.bert_word2id.items()}
        self.epoch = 0

        self.save_name = args.save_name


    def set_data_path(self, data_path):
        self.data_path = data_path
    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length
    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
    def set_window_size(self, window_size):
        self.window_size = window_size
    def set_word_size(self, word_size):
        self.word_size = word_size


    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_opt_method(self, opt_method):
        self.opt_method = opt_method
    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch
    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
    def set_is_training(self, is_training):
        self.is_training = is_training
    def set_use_bag(self, use_bag):
        self.use_bag = use_bag
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def set_meta_data(self, metadata):
        self.epoch = metadata['epoch'] + 1

    def get_meta_data(self):
        metadata = {"epoch": self.epoch}
        return metadata

    def set_publish(self):
        self.test_prefix = "dev_test"
        self.load_test_data()
    def set_train(self):
        self.test_prefix ="dev_dev"
        self.load_test_data()

    def _load_train_file(self):
        prefix = self.train_prefix
        dat_path = os.path.join(self.data_path, prefix + '.json')
        self.train_file = load_json(dat_path)

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix



        print ('train', prefix)
        dat_path =os.path.join(self.data_path, prefix+'.json')

        #q = mp.Queue()
        #p = mp.Process(target = load_json, args=(dat_path,q))
        #p.start()
        t = threading.Thread(target = self._load_train_file)
        t.start()

        self.word2id = json.load(open(os.path.join(self.data_path, 'word2id.json')))
        self.id2words = {v: k for k, v in self.word2id.items()}
        self.data_train_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
        self.data_train_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
        #self.train_file = load_json(dat_path)


        if self.use_bert:
            self.data_train_bert_word = np.load(os.path.join(self.data_path, prefix+'_bert_word.npy'))
            #self.data_train_bert_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
            self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
            self.data_train_bert_starts = np.load(os.path.join(self.data_path, prefix+'_bert_starts.npy'))

        #self.train_file = q.get()
        #p.join()
        t.join()
        print("Finish reading")

        self.train_len = ins_num = self.data_train_word.shape[0]
        print(self.train_len)
        assert(self.train_len==len(self.train_file))

        self.train_order = list(range(ins_num))
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1



    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}

        prefix = self.test_prefix
        print (prefix)
        self.is_test = ('dev_test' == prefix)
        self.data_test_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
        self.data_test_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        if self.use_bert:
            self.data_test_bert_word = np.load(os.path.join(self.data_path, prefix+'_bert_word.npy'))
            self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
            self.data_test_bert_starts = np.load(os.path.join(self.data_path, prefix+'_bert_starts.npy'))


        self.test_len = self.data_test_word.shape[0]
        assert(self.test_len==len(self.test_file))


        print("Finish reading")

        self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
        if self.data_test_word.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)


    def get_train_batch_mp(self, num_workers=3, queue_size=6):
        train_order = self.get_train_order()
        batch_ids = list(range(self.train_batches))

        t_begin_init = datetime.datetime.now()
        manager = Manager()
        queue = manager.Queue(queue_size)
        procs = []

        for wid in range(num_workers):
            print(f"Worker {wid+1}/{num_workers}")
            batch_ids_w = batch_ids[wid::num_workers]
            args = (wid, batch_ids_w, train_order, queue)
            p = Process(target=self._get_train_batch_w, args=args)
            p.start()
            print(f"start")
            procs.append(p)

        finish_count = 0
        batch_count = 0
        while finish_count < num_workers:
            batch = queue.get()
            batch_count += 1
            if batch is None:
                finish_count += 1
            else:
                if batch['torch']:
                    yield batch
                else:
                    yield self.batch_from_np2torch(batch)

    def _get_train_batch_w(self, worker_id, batch_ids, train_order, queue):
        #print("worker ",worker_id," start")
        for b in batch_ids:
            print("BC data-begin", b)
            batch = self._get_train_batch(b, train_order)
            queue.put(batch)
            print("BC data-end", b)
            #break

        queue.put(None)

    def get_train_batch_debug(self):
        train_order = self.get_train_order()
        batch_ids = list(range(self.train_batches))

        for b in batch_ids:
            batch = self._get_train_batch(b, train_order)
            if batch['torch']:
                yield batch
            else:
                yield self.batch_from_np2torch(batch)

    def get_test_order(self):
        return list(range(len(self.test_file)))


    def get_test_batch_debug(self):
        test_order = self.get_test_order()
        batch_ids = list(range(self.test_batches))

        for b in batch_ids:
            batch = self._get_test_batch(b, test_order)
            if batch['torch']:
                yield batch
            else:
                yield self.batch_from_np2torch(batch)



    def batch_from_np2torch(self, batch):
        if batch['torch']:
            return batch

        to_cuda = True
        if "to_cuda" in batch and not batch["to_cuda"]:
            to_cuda = False

        for key in self.batch_keys:
            if key in batch:
                try:
                    if to_cuda:
                        batch[key] = torch.from_numpy(batch[key]).cuda().contiguous()
                    else:
                        batch[key] = torch.from_numpy(batch[key])
                except Exception as e:
                    print(key)
                    raise e


        for key in self.batch_keys_float:
            if key in batch:
                if HALF_PRECISION:
                    batch[key] = torch.from_numpy(batch[key]).half()
                else:
                    batch[key] = torch.from_numpy(batch[key]).float()
                if to_cuda:
                    batch[key] = batch[key].cuda().contiguous()

        return batch

    def save_model(self, model, optimizer, best=False):
        if best:
            model_fname = f"{self.checkpoint_dir}/{self.save_name}_best.chp"
        else:
            model_fname = f"{self.checkpoint_dir}/{self.save_name}_epoch{self.epoch}.chp"
        torch.save(model.state_dict(), model_fname)

        optimizer_fname = model_fname + ".opt"
        torch.save(optimizer.state_dict(), optimizer_fname)

        meta_fname = model_fname + ".dat"
        with open(meta_fname, "wb") as f:
            pickle.dump(self.get_meta_data(), f)


    def load_model_optimizer(self, model_pattern, pretrain_model = None):
        ori_model = model_pattern(config=self)
        if HALF_PRECISION:
            ori_model.half()
        if pretrain_model is not None:
            self.pretrain_model = pretrain_model

        load_model = self.pretrain_model and "epoch0" not in self.pretrain_model
        load_opt = load_model

        #'''
        model = nn.DataParallel(ori_model)
        #model = ori_model
        model.cuda()
        #model = ori_model
        if load_model:
            model = self.load_model(model, self.pretrain_model)
            metadata = self.load_metadata()
            self.set_meta_data(metadata)


        lr = self.learning_rate
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        if load_opt:
            optimizer = self.load_optimizer(optimizer, self.pretrain_model)

        return model, optimizer

    def load_model(self, model, pretrain_model=""):
        model_fname = f"{self.checkpoint_dir}/{self.pretrain_model}.chp"
        state_dict = torch.load(model_fname)
        model.load_state_dict(state_dict)
        return model

    def load_optimizer(self, optimizer, pretrain_model):
        model_fname = f"{self.checkpoint_dir}/{self.pretrain_model}.chp"
        optimizer_fname = model_fname + ".opt"
        state_dict = torch.load(optimizer_fname)
        optimizer.load_state_dict(state_dict)
        return optimizer

    def load_metadata(self):
        fname = f"{self.checkpoint_dir}/{self.pretrain_model}.chp.dat"
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                metadata = pickle.load(f)
        else:
            metadata = {}
        return metadata


