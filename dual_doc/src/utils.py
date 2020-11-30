import json

import numpy as np
import networkx as nx
import multiprocessing as mp
def descobj(obj):
    if isinstance(obj, dict):
        for k in obj.keys():
            print(k)
            desc(obj[k], True)
    else:
        for k in obj.__dict__.keys():
            print(k)
            desc(getattr(obj, k), True)

def argdict2str(filename, argdict):
    cmd_str = f"python {filename} "
    for key, val in argdict.items():
        cmd_str += f"--{key} {val} "
    return cmd_str

def desc(obj, short=False):
    if isinstance(obj, list):
        print("List", f"Len: {len(obj)}")
    elif isinstance(obj, dict):
        print("Dict", f"Len: {len(obj)}")
    elif isinstance(obj, np.ndarray):
        print(f"numpy ndarray: {obj.shape}")
    elif isinstance(obj, int):
        print("Int", obj)
    elif isinstance(obj, str):
        print("Str", obj)
    else:
        if short:
            print(type(obj))
        else:
            help(obj)
            dir(obj)

class dict_set(dict):
    def __init__(self):
        super().__init__()

    def add(self, key, val):
        if key not in self:
            self[key] = set()
        self[key].add(val)

    def size(self, key):
        if key not in self:
            return 0
        return len(self[key])




class dict_list(dict):
    def __init__(self):
        super().__init__()

    def add_list(self, key, val_list):
        if key in self:
            self[key] += val_list
        else:
            self[key] = val_list

    def add(self, key, val):
        if key in self:
            self[key].append(val)
        else:
            self[key] = [val]

    def remove_duplicate(self):
        for key, val_list in self.items():
            self[key] = list(set(val_list))

class dict_max(dict):
    def __init__(self):
        super().__init__()

    def add(self, key, val):
        if key in self:
            self[key] = max(val, self[key])
        else:
            self[key] = val

def print_label_abbr(label, g = None, i = -1, id2rel = None, reldata = None):
    h = label["h"]
    t = label["t"]
    r = label["r"]
    evidence = label["evidence"]

    if i >= 0:
        prefix = f"{i}\t"
    else:
        prefix = ""

    print(f"{prefix} <{h},{r},{t}>", evidence)
    if id2rel is not None:
        rel = id2rel[r]
        print(f"{prefix} <{h},{rel},{t}>", evidence)


    if g is not None:
        g.add_edge(h,t, reltype = r)



def print_labels_abbr(labels, id2rel = None, reldata = None):
    g = nx.DiGraph()
    for i, label in enumerate(labels):
        print_label_abbr(label, g = g, i = i, id2rel = id2rel, reldata = reldata)
    return g


def print_label_txt(labels, vertexSet, id2rel, id2relname, min_dist_mid = None):
    for label in labels:
        _print_label_txt(label, vertexSet, id2rel, id2relname, min_dist_mid)

def _print_label_txt(label, vertexSet, id2rel, id2relname, min_dist_mid = None):
    h, r, t = label['h'], label['r'], label['t']
    relname = id2relname[r]


    if min_dist_mid is None:
        h_em = vertexSet[h][0]
        t_em = vertexSet[t][0]
        str_basic = f"<{h_em['name']}, ({r},{relname}), {t_em['name']}>\t{label['evidence']}"
        print(str_basic)
    else:
        m_h, m_t = min_dist_mid[h,t]
        h_em = vertexSet[h][m_h]
        t_em = vertexSet[t][m_t]
        str_basic = f"<{h_em['name']}, ({r},{relname}), {t_em['name']}>   [{h_em['sent_id']},{t_em['sent_id']}]\t{label['evidence']}"
        print(str_basic)



class Accuracy():
    def __init__(self, n_tp = 0, n_pos = 0, n_labels = 0):
        self.n_tp = n_tp
        self.n_pos = n_pos
        self.n_labels = n_labels

    def get_result(self):
        if self.n_tp == 0:
            return 0.0, 0.0, 0.0

        precision = 1.0 * self.n_tp / self.n_pos
        recall = 1.0 * self.n_tp / self.n_labels
        f1 = 2.0 * precision * recall / (precision + recall)
        return precision, recall, f1

    def inc_tp(self, n = 1):
        self.n_tp += n
    def inc_pos(self, n = 1):
        self.n_pos += n
    def inc_labels(self, n = 1):
        self.n_labels += n

    def print_head(self, name = False):
        if name:
            tab = f"{name:12s}\t"
        else:
            tab = ""
        print(f"{tab}{'#TP':8s}\t{'#Pos':8s}\t{'#Labels':8s}\t{'Precision':8s}\t{'Recall':8s}\t{'F1':8s}")

    def print_result(self, print_head = True, name = ""):
        precision, recall, f1 = self.get_result()
        if print_head:
            self.print_head(len(name) >0)

        if len(name) >0:
            name = f"{name:12s}\t"
        print(name+f"{self.n_tp:8d}\t{self.n_pos:8d}\t{self.n_labels:8d}\t{precision:.6f}\t{recall:.6f}\t{f1:.6f}")

    def __add__(self, other):
        acc = Accuracy()
        for acc_tmp in [self, other]:
            acc.inc_labels(acc_tmp.n_labels)
            acc.inc_tp(acc_tmp.n_tp)
            acc.inc_pos(acc_tmp.n_pos)
        return acc

    def __sub__(self, other):
        return AccuracyDelta(self, other)

class AccuracyDelta():
    def __init__(self, accu1, accu2):
        assert accu1.n_labels == accu2.n_labels
        self.n_labels = accu1.n_labels
        self.n_tp_diff = accu1.n_tp - accu2.n_tp
        self.n_pos_diff = accu1.n_pos - accu2.n_pos

        res1 = accu1.get_result()
        res2 = accu2.get_result()
        self.f1_diff = res1[2] - res2[2]
        self.precision_diff = res1[0] - res2[0]
        self.recall_diff = res1[1] - res2[1]

        self.res1 = (accu1.n_tp, accu1.n_pos, accu1.n_labels) + res1
        self.res2 = (accu2.n_tp, accu2.n_pos, accu2.n_labels) + res2

    def print(self):
        print(f"{'-':8s}\t{'#TP':8s}\t{'#Pos':8s}\t{'#Labels':8s}\t{'Precision':8s}\t{'Recall':8s}\t{'F1':8s}")
        print(f"{'Before':8s}\t{self.res2[0]:8d}\t{self.res2[1]:8d}\t{self.res2[2]:8d}\t{self.res2[3]:.6f}\t{self.res2[4]:.6f}\t{self.res2[5]:.6f}")
        print(f"{'After':8s}\t{self.res1[0]:8d}\t{self.res1[1]:8d}\t{self.res1[2]:8d}\t{self.res1[3]:.6f}\t{self.res1[4]:.6f}\t{self.res1[5]:.6f}")
        print(f"{'Change':8s}\t{self.res1[0]-self.res2[0]:8d}\t{self.res1[1]-self.res2[1]:8d}\t{'-':8s}\t{self.res1[3]-self.res2[3]:.6f}\t{self.res1[4]-self.res2[4]:.7f}\t{self.res1[5]-self.res2[5]:.6f}")


def t_info(dat):
    print(f"Shape: {dat.shape}, Mean: {dat.mean()}, Std: {dat.std()}, Min: {dat.min()}, Max: {dat.max()}")

def stack_if_islist(dat):
    if isinstance(dat, list):
        return

def load_json(path, q = None):
    dat = json.load(open(path))
    #dat = {}
    if isinstance(q, mp.queues.Queue):
        print("Put")
        q.put(dat)
    return dat

def from_cpu_to_cuda(t, device):
    return t.cuda(device).contiguous()

def from_cpu_to_cuda_list(tensor_list, device):
    output = [from_cpu_to_cuda(t,device) if t is not None else None for t in tensor_list]
    return output