import os

import config, models
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
from config.HDTuneConfig import TUNING_OPTION_BIAS_SET, TUNING_OPTION_BIAS_FIX
from train import get_ext_parser

parser = get_ext_parser()
parser.add_argument('--input_theta', type = float, default = -1)
parser.add_argument('--secondary_stat', type = bool, default = False)



model_dict = {
    'CNN3': models.CNN3,
    'LSTM': models.LSTM,
    'BiLSTM': models.BiLSTM,
    'ContextAware': models.ContextAware,
    'BERT':models.Bert_Ext,
    "BERTB":models.BertBasicExt,
    'DualBERT':models.DualExt,
    models.DualExtAdaptive.name: models.DualExtAdaptive,
    models.DualExtAdaptiveLight.name : models.DualExtAdaptiveLight,
    models.BERT_RE.name : models.BERT_RE,
    models.DualBERTStacked.name: models.DualBERTStacked,
    models.DualBERTSkip.name: models.DualBERTSkip
}


if __name__ == "__main__":
    args = parser.parse_args()

    if args.config == config.HDTuneConfig.name:
        con = config.HDTuneConfig(args)

    model_pattern = model_dict[args.model_name]
    if args.save_name :
        save_name_base = args.save_name
    else:
        save_name_base = con.get_expname(model_pattern)

    secondary_stat = False

    pretrain_model = f"{save_name_base}_best"
    print(pretrain_model)

    if os.path.exists(f"{con.checkpoint_dir}/{pretrain_model}.chp"):
        con.load_test_data()
        con.load_train_data()

        model, optimizer = con.load_model_optimizer(model_pattern, pretrain_model)
        save_name = f"{save_name_base}"
        model.eval()

        if args.tuning_opt == TUNING_OPTION_BIAS_SET:
            f1, auc, pr_x, pr_y, theta = con.test(model, save = True)
            print(f"Set-bias F1: {f1}, Theta: {theta}")
            con.publish(model, save_name, theta)

            con.set_publish()
            con.publish(model, save_name, theta)
        elif args.tuning_opt == TUNING_OPTION_BIAS_FIX:
            f1, auc, pr_x, pr_y, theta_original = con.test(model, save=True, original=True)
            print(f"Original F1: {f1}, Theta: {theta_original}")
            con.publish(model, save_name, theta_original, original= True)

            f1, auc, pr_x, pr_y, theta_fb = con.test(model, save=True)
            print(f"Fix-bias F1: {f1}, Theta: {theta_fb}")
            con.publish(model, save_name, theta_fb)


            con.set_publish()
            con.publish(model, save_name, theta_original, original = True)
            con.publish(model, save_name, theta_fb)



