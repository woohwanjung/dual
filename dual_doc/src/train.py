import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
import random
from datetime import datetime

import torch

import config
import models
from config import DualConfig
from config.HDTuneConfig import TUNING_OPTION_BIAS_SET


def get_ext_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
    parser.add_argument('--save_name', type = str)

    parser.add_argument('--train_prefix', type = str, default = 'dev_train')
    parser.add_argument('--test_prefix', type = str, default = 'dev_dev')

    parser.add_argument("--hidden_size", type = int, default = 128)
    parser.add_argument("--batch_size", type = int, default = 40)
    parser.add_argument("--use_bert", type = bool, default = False)
    #parser.add_argument("--train_bert", type = bool, default = False)

    parser.add_argument("--debug", type = bool, default = False)
    parser.add_argument("--max_epoch", type = int, default = 200)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--lr_shiftdown_epoch", type = int, default = 100)
    parser.add_argument("--lr_shiftdown",type = float, default = 0.1)
    parser.add_argument("--w_dist", type = float, default = 0.0001)
    parser.add_argument("--force_continue", type = bool, default = False)


    #Option Dual
    parser.add_argument("--dualopt",type=int, default = 0)
    parser.add_argument("--mu_activation", type = str, default = "tanh")
    parser.add_argument("--distance", type = int, default = DualConfig.DISTANCE_DIFF_GAUSSIAN)
    parser.add_argument("--negativelabel", type = int, default = DualConfig.NEGATIVE_ALL)
    parser.add_argument("--ner_emb", type = bool, default = False)

    #Option Dual Shuffle
    parser.add_argument("--proportion_ha",type = float, default = 0.5)
    parser.add_argument("--hatrain_partial",type = float, default= 1.0)
    parser.add_argument("--sanity_bound", type = float, default = 1.0)
    parser.add_argument("--reg_mean", type = bool, default = False)
    parser.add_argument("--reg_std", type = bool, default = False)
    parser.add_argument("--cross_encoder", type = bool, default = False)
    parser.add_argument("--num_cross_encoders", type = int, default = 1)
    parser.add_argument("--skip_connection",type=int, default = 1)
    parser.add_argument("--zeromean_epoch", type = int, default = 0)
    parser.add_argument("--sb_prob", type = float, default = 0.0)
    parser.add_argument("--diff_label", type = bool, default = False)

    parser.add_argument("--augha", type = bool, default = False)
    parser.add_argument("--diffonins", type = bool, default = False)
    parser.add_argument("--difflogit", type = bool, default = False)
    parser.add_argument("--train_nonerel", type = bool, default = False)
    parser.add_argument("--additional_opt", type = str, default = "")

    #Option HDTune
    parser.add_argument("--tuning_opt", type=int, default=TUNING_OPTION_BIAS_SET)


    #parser.add_argument("--")

    parser.add_argument("--config", type=str, default = "dualshuffle")
    parser.add_argument("--num_output_module", type= int, default = 1)
    parser.add_argument("--twin_init", type = bool, default = False)
    parser.add_argument("--light", type = bool, default = False)
    parser.add_argument("--const_mu", type = float, default = -1.0)

    parser.add_argument("--reduce", type = int, default = DualConfig.LOSS_REDUCTION_SUM)
    parser.add_argument("--random_seed",type = int, default = 0)

    return parser

model_dict = {
        'CNN3': models.CNN3,
        'LSTM': models.LSTM,
        'BiLSTM': models.BiLSTM,
        'ContextAware': models.ContextAware,
        models.BERT_RE.name : models.BERT_RE,
    }

if __name__ == "__main__":
    print(datetime.now())
    parser = get_ext_parser()
    args = parser.parse_args()

    torch.cuda.empty_cache()
    random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    if args.config == "dual":
        con = config.DualConfig(args)
    elif args.config == "baseline":
        con = config.Config(args)
    elif args.config == "dualinterleave":
        con = config.DualConfigInterleave(args)
    elif args.config == config.HDTuneConfig.name:
        con = config.HDTuneConfig(args)

    #con = config.Config(args)
    #con = config.ConfigB(args)
    #con = config.DualConfig(args)
    con.set_max_epoch(args.max_epoch)
    con.load_train_data()
    con.load_test_data()
    # con.set_train_model()
    con.train(model_dict[args.model_name])
