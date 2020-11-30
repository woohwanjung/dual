import argparse
import os
import random
import sys

from config.DualConfig import get_expname
from config.HDTuneConfig import TUNING_OPTION_BIAS_FIX, TUNING_OPTION_BIAS_SET
from models.bert import BERT_RE
from settings import *
from config import DualConfig
from models import CNN3, BiLSTM, LSTM, ContextAware

def get_cmd_from_args(filename, argdict, cuda_devices = None):
    cmd = f"python {filename} "
    if cuda_devices is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} " + cmd
    for key, val in argdict.items():
        if val or not isinstance(val, bool):
            cmd += f"--{key} {val} "

    return cmd


def test_pass_exp(explog_set, argdict, train = True):
    opt_str = get_expname(argdict)
    print(opt_str)

    done = False
    for lf in explog_set:
        if lf.endswith(opt_str):
            done = True
            break
    if done:
        if train:
            print("=Done", opt_str)
            return True
    else:
        if not train:
            print("No result", opt_str)
            return True
    return False


def get_exp_cmd_tune(train = True, test = True, cuda_devices = None):
    filename_train = "train.py"
    filename_test = "test_tune.py"
    config_key = "hdtune"

    bert_model_names = [BERT_RE.name]
    basic_model_names = [CNN3.name, BiLSTM.name, LSTM.name, ContextAware.name]


    hidden_size = 128
    batch_size_bert = 12
    batch_size = 40

    num_output_module = 1
    max_epoch = 10
    lr_bert = 1e-5
    lr_basic = 1e-3

    train_none_rel_list = [True]
    reduce_list = [DualConfig.LOSS_REDUCTION_MEAN]
    negative_option = [DualConfig.NEGATIVE_3TIMES]
    negative_option_basic = [DualConfig.NEGATIVE_ALL]


    cross_entity_encoder_list = [False]
    use_ner_emb_list = [False]
    opt_bias_list = [TUNING_OPTION_BIAS_SET, TUNING_OPTION_BIAS_FIX]
    explog_set = os.listdir(f"{WRITER_DIR_DUAL}")

    for reduce in reduce_list:
        for cross_encoder in cross_entity_encoder_list:
            for opt_bias in opt_bias_list:
                for train_none_rel in train_none_rel_list:
                    argdict_base = {"config": config_key,
                                    "max_epoch": max_epoch,
                                    "reduce": reduce,
                                    "train_nonerel": train_none_rel,
                                    "hidden_size": hidden_size,
                                    "cross_encoder": cross_encoder,
                                    "tuning_opt": opt_bias,
                                    }
                    for negative_opt in negative_option:
                        #Without Dual
                        for model_name_bert in bert_model_names:
                            for ner_emb in use_ner_emb_list:
                                argdict = argdict_base.copy()
                                argdict["negativelabel"] = negative_opt
                                argdict["model_name"] = model_name_bert
                                argdict["batch_size"] = batch_size_bert
                                argdict["ner_emb"] = ner_emb
                                argdict["use_bert"] = True
                                argdict["learning_rate"] = lr_bert
                                if train:
                                    if test_pass_exp(explog_set, argdict, train=True):
                                        continue
                                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                                if test:
                                    if not train and test_pass_exp(explog_set, argdict, train=False):
                                        continue
                                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

                    for negative_opt in negative_option_basic:
                        for model_name_basic in basic_model_names:
                            argdict = argdict_base.copy()
                            argdict["model_name"] = model_name_basic
                            argdict["batch_size"] = batch_size
                            argdict["negativelabel"] = negative_opt
                            argdict["ner_emb"] = True
                            if model_name_basic == "CNN3":
                                argdict["hidden_size"] = 200

                            argdict["learning_rate"] = lr_basic
                            if train:
                                if test_pass_exp(explog_set, argdict, train=True):
                                    continue
                                yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                            if test:
                                if not train and test_pass_exp(explog_set, argdict, train=False):
                                    continue
                                yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)
def get_exp_cmd_ha_only(train = True, test = True, cuda_devices = None):
    filename_train = "train.py"
    filename_test = "test.py"
    config_key = "dualshuffle"

    bert_model_names = [BERT_RE.name]
    basic_model_names = [CNN3.name, BiLSTM.name, LSTM.name, ContextAware.name]

    hidden_size = 128
    batch_size_bert = 12
    batch_size_basic = 40

    max_epoch = 100
    lr_bert = 1e-5
    lr_basic = 1e-3


    mu_activation = "tanh"
    proportion_ha = 1.0
    sanity_bound = 0.0001
    sb_prob = 0.01
    augha = False
    difflogit = False

    lr = 1e-5
    diffonins = True
    train_none_rel = True

    reduce = DualConfig.LOSS_REDUCTION_MEAN
    negative_opt = DualConfig.NEGATIVE_3TIMES

    cross_entity_encoder_list = [False, True]
    cross_entity_encoder_list = [False]

    reg_mean = False
    reg_std = False
    w_dist = 1e-5

    explog_set = os.listdir(f"{WRITER_DIR_DUAL}")

    for cross_entity_encoder in cross_entity_encoder_list:
        argdict_base = {
                        "config":config_key,
                        "max_epoch": max_epoch,
                        "proportion_ha": proportion_ha,
                        "cross_encoder":cross_entity_encoder,
                        "reduce":reduce,
                        "num_output_module": 1,
                        "hidden_size": hidden_size,
                        "train_nonerel":train_none_rel,
                        "negativelabel":negative_opt}


        argdict_single_base = argdict_base.copy()
        argdict_single_base.update({})

        for model in bert_model_names:
            argdict = argdict_single_base.copy()
            argdict.update({
                "model_name": model,
                "learning_rate":lr_bert,
                "batch_size": batch_size_bert,
                "use_bert": True
            })
            if train and not test_pass_exp(explog_set, argdict, train=True):
                yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
            if test:
                yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

        for model in basic_model_names:
            argdict = argdict_single_base.copy()
            argdict.update({
                "model_name": model,
                "learning_rate": lr_basic,
                "batch_size": batch_size_basic,
            })
            if "CNN" in model:
                argdict['hidden_size'] = 200

            if train and not test_pass_exp(explog_set, argdict, train=True):
                yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
            if test:
                yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)



def get_exp_cmd_shuffle(train = True, test = True, cuda_devices = None, num_output_modules = 1):
    filename_train = "train.py"
    filename_test = "test.py"
    config_key = "dualshuffle"

    bert_model_names = [BERT_RE.name]
    basic_model_names = [CNN3.name, BiLSTM.name, LSTM.name, ContextAware.name]

    hidden_size = 128
    batch_size_bert = 12
    batch_size_basic = 40

    max_epoch = 100
    lr_bert = 1e-5
    lr_basic = 1e-3


    mu_activation = "tanh"
    proportion_ha = 0.5
    sanity_bound = 0.0001
    sb_prob = 0.01
    augha = False
    difflogit = False

    lr = 1e-5
    diffonins = True
    train_none_rel = True

    reduce = DualConfig.LOSS_REDUCTION_MEAN
    negative_opt = DualConfig.NEGATIVE_3TIMES

    cross_entity_encoder_list = [False, True]
    cross_entity_encoder_list = [False]

    reg_mean = False
    reg_std = False
    w_dist = 1e-5

    explog_set = os.listdir(f"{WRITER_DIR_DUAL}")

    for cross_entity_encoder in cross_entity_encoder_list:
        argdict_base = {
                        "config":config_key,
                        "max_epoch": max_epoch,
                        "proportion_ha": proportion_ha,
                        "cross_encoder":cross_entity_encoder,
                        "reduce":reduce,
                        "num_output_module": num_output_modules,
                        "hidden_size": hidden_size,
                        "train_nonerel":train_none_rel,
                        "negativelabel":negative_opt}


        if num_output_modules == 1:
            argdict_single_base = argdict_base.copy()
            argdict_single_base.update({})

            for model in bert_model_names:
                argdict = argdict_single_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_single_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

        elif num_output_modules == 2:
            argdict_mt_base = argdict_base.copy()
            argdict_mt_base.update({
                "distance": DualConfig.DISTANCE_NO
            })
            for model in bert_model_names:
                argdict = argdict_mt_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True,
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_mt_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)
        elif num_output_modules == 4:
            argdict_dual_base = argdict_base.copy()
            argdict_dual_base.update({
                "proportion_ha": proportion_ha,
                "sanity_bound": sanity_bound,
                "diffonins": diffonins,
                "difflogit": difflogit,
                "twin_init": True,
                "reg_mean": reg_mean,
                "reg_std": reg_std,
                "mu_activation": mu_activation,
                "distance": DualConfig.DISTANCE_INFLATION_LOGNORMAL,
                "w_dist":w_dist,
            })
            for model in bert_model_names:
                argdict = argdict_dual_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_dual_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)


def get_exp_cmd_shuffle(train = True, test = True, cuda_devices = None, num_output_modules = 1):
    filename_train = "train.py"
    filename_test = "test.py"
    config_key = "dual"

    bert_model_names = [BERT_RE.name]
    basic_model_names = [CNN3.name, BiLSTM.name, LSTM.name, ContextAware.name]
    basic_model_names = [CNN3.name]

    hidden_size = 128
    batch_size_bert = 12
    batch_size_basic = 40

    max_epoch = 100
    lr_bert = 1e-5
    lr_basic = 1e-3


    mu_activation = "tanh"
    proportion_ha = 0.5
    sanity_bound = 0.0001
    sb_prob = 0.01
    augha = False
    difflogit = False

    lr = 1e-5
    diffonins = True
    train_none_rel = True

    reduce = DualConfig.LOSS_REDUCTION_MEAN
    negative_opt = DualConfig.NEGATIVE_3TIMES
    negative_opt = DualConfig.NEGATIVE_ALL

    cross_entity_encoder_list = [False, True]
    cross_entity_encoder_list = [False]

    reg_mean = False
    reg_std = False
    w_dist = 1e-5

    explog_set = os.listdir(f"{WRITER_DIR_DUAL}")

    for cross_entity_encoder in cross_entity_encoder_list:
        argdict_base = {
                        "config":config_key,
                        "max_epoch": max_epoch,
                        "proportion_ha": proportion_ha,
                        "cross_encoder":cross_entity_encoder,
                        "reduce":reduce,
                        "num_output_module": num_output_modules,
                        "hidden_size": hidden_size,
                        "train_nonerel":train_none_rel,
                        "negativelabel":negative_opt}


        if num_output_modules == 1:
            argdict_single_base = argdict_base.copy()
            argdict_single_base.update({})

            for model in bert_model_names:
                argdict = argdict_single_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_single_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

        elif num_output_modules == 2:
            argdict_mt_base = argdict_base.copy()
            argdict_mt_base.update({
                "distance": DualConfig.DISTANCE_NO
            })
            for model in bert_model_names:
                argdict = argdict_mt_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True,
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_mt_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)
        elif num_output_modules == 4:
            argdict_dual_base = argdict_base.copy()
            argdict_dual_base.update({
                "proportion_ha": proportion_ha,
                "sanity_bound": sanity_bound,
                "diffonins": diffonins,
                "difflogit": difflogit,
                "twin_init": True,
                "reg_mean": reg_mean,
                "reg_std": reg_std,
                "mu_activation": mu_activation,
                "distance": DualConfig.DISTANCE_INFLATION_LOGNORMAL,
                "w_dist":w_dist,
            })
            for model in bert_model_names:
                argdict = argdict_dual_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_dual_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)



def get_exp_cmd_varying_ha(train = True, test = True, cuda_devices = None, hatrain_partial = 0.1, num_output_modules = 1):
    filename_train = "train.py"
    filename_test = "test.py"
    config_key = "dualshuffle"

    bert_model_names = [BERT_RE.name]
    basic_model_names = [CNN3.name, BiLSTM.name, LSTM.name, ContextAware.name]
    basic_model_names = [BiLSTM.name]


    hidden_size = 128
    batch_size_bert = 12
    batch_size_basic = 40

    max_epoch = 100
    lr_bert = 1e-5
    lr_basic = 1e-3


    mu_activation = "tanh"
    proportion_ha = 0.5
    sanity_bound = 0.0001
    sb_prob = 0.01
    augha = False
    difflogit = False

    lr = 1e-5
    diffonins = True
    train_none_rel = True

    reduce = DualConfig.LOSS_REDUCTION_MEAN
    negative_opt = DualConfig.NEGATIVE_3TIMES
    negative_opt = DualConfig.NEGATIVE_ALL

    cross_entity_encoder_list = [False, True]
    cross_entity_encoder_list = [False]

    reg_mean = False
    reg_std = False
    w_dist = 1e-5

    explog_set = os.listdir(f"{WRITER_DIR_DUAL}")

    for cross_entity_encoder in cross_entity_encoder_list:
        argdict_base = {"config":config_key,
                        "max_epoch": max_epoch,
                        "proportion_ha": proportion_ha,
                        "hatrain_partial":hatrain_partial,
                        "cross_encoder":cross_entity_encoder,
                        "reduce":reduce,
                        "num_output_module": num_output_modules,
                        "hidden_size": hidden_size,
                        "train_nonerel":train_none_rel,
                        "negativelabel":negative_opt,
                        "hatrain_partial":hatrain_partial}


        if num_output_modules == 1:
            argdict_single_base = argdict_base.copy()
            argdict_single_base.update({})

            for model in bert_model_names:
                argdict = argdict_single_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_single_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

        elif num_output_modules == 2:
            argdict_mt_base = argdict_base.copy()
            argdict_mt_base.update({
                "distance": DualConfig.DISTANCE_NO
            })
            for model in bert_model_names:
                argdict = argdict_mt_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True,
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_mt_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)
        elif num_output_modules == 4:
            argdict_dual_base = argdict_base.copy()
            argdict_dual_base.update({
                "proportion_ha": proportion_ha,
                "sanity_bound": sanity_bound,
                "diffonins": diffonins,
                "difflogit": difflogit,
                "twin_init": True,
                "reg_mean": reg_mean,
                "reg_std": reg_std,
                "mu_activation": mu_activation,
                "distance": DualConfig.DISTANCE_INFLATION_LOGNORMAL,
                "w_dist":w_dist,
            })
            for model in bert_model_names:
                argdict = argdict_dual_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate":lr_bert,
                    "batch_size": batch_size_bert,
                    "use_bert": True
                })
                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)

            for model in basic_model_names:
                argdict = argdict_dual_base.copy()
                argdict.update({
                    "model_name": model,
                    "learning_rate": lr_basic,
                    "batch_size": batch_size_basic,
                })
                if "CNN" in model:
                    argdict['hidden_size'] = 200

                if train and not test_pass_exp(explog_set, argdict, train=True):
                    yield get_cmd_from_args(filename_train, argdict, cuda_devices=cuda_devices)
                if test:
                    yield get_cmd_from_args(filename_test, argdict, cuda_devices=cuda_devices)



parser = argparse.ArgumentParser()
parser.add_argument("--cuda_devices", type = str, default = None)
parser.add_argument("--exp_tune", type = bool, default = False)
parser.add_argument("--exp_shuffle", type = bool, default = False)
parser.add_argument("--exp_varyingha", type = bool, default = False)
parser.add_argument("--exp_haonly", type = bool, default = False)
parser.add_argument("--train", type = bool, default = False)
parser.add_argument("--test", type = bool, default = False)
parser.add_argument("--run", type = bool, default = False)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    cuda_devices = args.cuda_devices

    cmd_list = []
    if args.exp_tune:
        cmd_list += list(get_exp_cmd_tune(train = args.train, test = args.test, cuda_devices = cuda_devices))
    if args.exp_haonly:
        cmd_list += list(
            get_exp_cmd_ha_only(train=args.train, test=args.test, cuda_devices=cuda_devices))
    if args.exp_shuffle:
        cmd_list += list(get_exp_cmd_shuffle(train = args.train, test = args.test, cuda_devices = cuda_devices, num_output_modules= 1))
        cmd_list += list(get_exp_cmd_shuffle(train = args.train, test = args.test, cuda_devices = cuda_devices, num_output_modules= 2))
        cmd_list += list(get_exp_cmd_shuffle(train = args.train, test = args.test, cuda_devices = cuda_devices, num_output_modules= 4))
    if args.exp_varyingha:
        #'''
        cmd_list += list(get_exp_cmd_varying_ha(train = args.train, test = args.test, hatrain_partial = 0.1, num_output_modules= 4, cuda_devices = cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train = args.train, test = args.test, hatrain_partial = 0.1, num_output_modules= 2, cuda_devices = cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train = args.train, test = args.test, hatrain_partial = 0.1, num_output_modules= 1, cuda_devices = cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.5, num_output_modules=4, cuda_devices=cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.5, num_output_modules=2, cuda_devices=cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.5, num_output_modules=1, cuda_devices=cuda_devices))

        cmd_list += list(get_exp_cmd_varying_ha(train = args.train, test = args.test, hatrain_partial = 0.25, num_output_modules= 4, cuda_devices = cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train = args.train, test = args.test, hatrain_partial = 0.25, num_output_modules= 2, cuda_devices = cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train = args.train, test = args.test, hatrain_partial = 0.25, num_output_modules= 1, cuda_devices = cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.75, num_output_modules=4, cuda_devices=cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.75, num_output_modules=2, cuda_devices=cuda_devices))
        cmd_list += list(get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.75, num_output_modules=1, cuda_devices=cuda_devices))
        # '''
        cmd_list += list(
            get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.15, num_output_modules=4,
                                   cuda_devices=cuda_devices))
        cmd_list += list(
            get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.15, num_output_modules=2,
                                   cuda_devices=cuda_devices))
        cmd_list += list(
            get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.15, num_output_modules=1,
                                   cuda_devices=cuda_devices))
        cmd_list += list(
            get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.05, num_output_modules=4,
                                   cuda_devices=cuda_devices))
        cmd_list += list(
            get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.05, num_output_modules=2,
                                   cuda_devices=cuda_devices))
        cmd_list += list(
            get_exp_cmd_varying_ha(train=args.train, test=args.test, hatrain_partial=0.05, num_output_modules=1,
                                   cuda_devices=cuda_devices))

    for i, cmd in enumerate(cmd_list):
        print(f"{i}/{len(cmd_list)}\t{cmd}")

    if args.run:
        for i, cmd in enumerate(cmd_list):
            print(f"{i}/{len(cmd_list)}\t{cmd}")
            os.system(cmd)