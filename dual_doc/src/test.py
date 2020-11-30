import os

import config, models
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
from train import get_ext_parser, model_dict

parser = get_ext_parser()
parser.add_argument('--input_theta', type = float, default = -1)




if __name__ == "__main__":
    args = parser.parse_args()

    if args.config == "dualshuffle":
        con = config.DualConfigShuffle(args)
    elif args.config == "dual":
        con = config.DualConfig(args)
    elif args.config == "baseline":
        con = config.Config(args)
    elif args.config == "dualinterleave":
        con = config.DualConfigInterleave(args)
    elif args.config == config.HDTuneConfig.name:
        con = config.HDTuneConfig(args)

    model_pattern = model_dict[args.model_name]
    if args.save_name:
        save_name_base = args.save_name
    else:
        save_name_base = con.get_expname(model_pattern)



    pretrain_model = f"{save_name_base}_best"
    print(pretrain_model)

    if os.path.exists(f"{con.checkpoint_dir}/{pretrain_model}.chp"):
        con.load_test_data()

        model, optimizer = con.load_model_optimizer(model_pattern, pretrain_model)
        save_name = f"{save_name_base}"
        model.eval()

        #print("Relation wise---")
        #th_by_relation, f1, f1_ignore, precision, precision_ignore, recall, recall_ignore = con.test_relationwise_th(model)
        #con.publish_rwt(model, f"{save_name}_rwt", th_by_relation)


        print("Single_threshold---")
        if args.input_theta > 0.0:
            f1, auc, pr_x, pr_y, theta = con.test(model, input_theta= args.input_theta)
        else:
            f1, auc, pr_x, pr_y, theta = con.test(model, save = True)
            pass
            #con.save_test_result(model,f1, auc, pr_x, pr_y, theta, accuracy_by_rel)
        print(f"F1: {f1}, Theta: {theta}")




        #con.publish(model, f"{save_name}_top1", -1.0)
        con.publish(model, save_name, theta)

        con.set_publish()
        #con.publish(model, f"{save_name}_0.5", 0.5)
        #con.publish(model, f"{save_name}_top1", -1.0)
        con.publish(model, save_name, theta)
        #con.publish_rwt(model, f"{save_name}_rwt", th_by_relation)


