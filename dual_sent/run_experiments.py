import argparse
import os


def get_cmd_from_args(filename, argdict, device_id = None):
    cmd = f"~/anaconda/envs/whjung/bin/python {filename} "
    if device_id is None:
        cmd = f"python {filename} "
    else:
        cmd = f"CUDA_VISIBLE_DEVICES={device_id} python {filename} "

    for key, val in argdict.items():
        if val is None:
            cmd += f" --{key} "
        elif val or not isinstance(val, bool):
            cmd += f" --{key} {val} "

    return cmd


def info_name(dataset, model, opt_dict):
    info = f"{dataset}_{model}"

    if opt_dict['optimizer'] != "SGD":
        info += f"_{opt_dict['optimizer']}"
    info += f"_lr{opt_dict['lr']}"
    if "weight_decay" in opt_dict and opt_dict["weight_decay"] >0:
        info += f"_w{opt_dict['weight_decay']}"

    if "dual" in opt_dict and opt_dict["dual"]:
        info += f"_dual_wd{opt_dict['w_dist']}"
        if "diffall" in opt_dict and opt_dict["diffall"]:
            info += "_da"
        if "crossiter" in opt_dict and opt_dict["crossiter"]:
            info += "_ci"
        return info
    elif "multitask" in opt_dict and opt_dict["multitask"]:
        info += f"_multitask"
        if "crossiter" in opt_dict and opt_dict["crossiter"]:
            info += "_ci"
        return info
    elif "union" in opt_dict and opt_dict["union"]:
        info += f"_union"
        if "crossiter" in opt_dict and opt_dict["crossiter"]:
            info += "_ci"
        return info
    elif "shuffle" in opt_dict and opt_dict["shuffle"]:
        info += f"_shuffle"
        return info
    elif "haonly" in opt_dict and opt_dict["haonly"]:
        info += f"_haonly"
        return info

    info += f"_fb{opt_dict['fix_bias']}"



    return info

def build_test_argdict(train_argdict):
    test_argdict = {}
    for k, v in train_argdict.items():
        if k in ["info","fix_bias","repeat"]:
            test_argdict[k] = v
    return test_argdict

def get_exp_command(train = True, device_id =None):
    use_bert_model = False

    if use_bert_model:
        filename = "Neural/train_bert.py" if train else "Neural/eva_bert.py"
    else:
        filename = "Neural/train.py" if train else "Neural/eva.py"


    hidden_bert = 128
    dataset_list = ["NYT","KBP"]

    if use_bert_model:
        model_list = ["sbert"]
        optimizer_list = ["Adam"]
    else:
        model_list = ["cnn", "pcnn", "bgru", "lstm", "pa_lstm"]
        optimizer_list = ["SGD"]



    dual_crossiter_list = [True]
    dual_w_dist_list = [1e-4]
    if use_bert_model:
        weight_decay = 1e-4
    else:
        weight_decay = 0.001
    repeats = 5

    fix_bias_list = [None, False]

    for dataset in dataset_list:
        for optimizer in optimizer_list:
            if optimizer == "SGD":
                lr_list = [1.0]
            elif optimizer == "Adam":
                lr_list = [1e-4]
            for lr in lr_list:
                for model in model_list:
                    argdict_base = {"model":model,
                                    "data_dir":f"data/neural/{dataset}",
                                    "optimizer": optimizer,
                                    "lr":lr,
                                    "repeat":repeats,
                                    "skip_ifexists": True}
                    if model ==  "sbert":
                        argdict_base['hidden'] = hidden_bert
                    if optimizer == "Adam":
                        argdict_base['weight_decay'] = weight_decay

                    #'''
                    for w_dist in dual_w_dist_list:
                        for crossiter in dual_crossiter_list:
                            argdict = argdict_base.copy()

                            argdict["dual"] = True
                            argdict["w_dist"] = w_dist
                            argdict["crossiter"] = crossiter

                            argdict["info"] = info_name(dataset, model, argdict)
                            print(argdict["info"])
                            if test:
                                argdict = build_test_argdict(argdict)
                            if model == "sbert" and train:
                                argdict["num_epoch"] = 10 if lr < 2e-5 else 5
                            yield get_cmd_from_args(filename, argdict)
                    #'''

                    #'''
                    for fix_bias in fix_bias_list:
                        argdict = argdict_base.copy()
                        argdict['fix_bias'] = fix_bias
                        argdict["info"] = info_name(dataset, model, argdict)
                        print(argdict["info"])
                        if test:
                            argdict = build_test_argdict(argdict)
    
                        if model == "sbert" and train:
                            argdict["num_epoch"] = 10 if lr < 2e-5 else 5
                        yield get_cmd_from_args(filename, argdict)
                    #'''


                    #'''
                    argdict = argdict_base.copy()
                    argdict["haonly"] = True
                    argdict["info"] = info_name(dataset, model, argdict)
                    print(argdict["info"])
                    if train:
                        argdict["num_epoch"] = 100
                    if test:
                        argdict = build_test_argdict(argdict)
                    yield get_cmd_from_args(filename, argdict)
                    #'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=False, help='run test')
    parser.add_argument('--cuda_visible_devices', type=str, default=None, help='gpu')
    args = parser.parse_args()

    test = args.test
    train = not test

    cmd_list = list(get_exp_command(train = train, device_id=args.cuda_visible_devices))
    for i, cmd_str in enumerate(cmd_list):
        print(i, cmd_str)
    for i, cmd_str in enumerate(cmd_list):
        print(i, cmd_str)
        #os.system(cmd_str)
        #break
