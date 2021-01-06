import os
import pickle as pkl
import numpy as np

F1_MAX = 1
def collect_result(dirpath ="eval_result", adam = False):
    res_dict = {}
    for fname in os.listdir(dirpath):
        with open(f"{dirpath}/{fname}", "rb") as f:
            res = pkl.load(f)
            info = fname[:-4]

            if "sbert" in info:#TODO: Hard coded
                info = info.replace("_Adam","")

            if "haonly" in info and "original" not in info:
                continue
            attrs = exp_attributs(info)
            if attrs.hd_alg == "dual":
                if "ddTrue" in attrs.hd_opt:
                    pass
                    continue

            if adam and "Adam" not in info:
                continue
            if not adam and "Adam" in info:
                continue

            res_dict[info] = (attrs,res)
    print(f"Read {len(res_dict)} result files")

    print(f"{'Data':40s}\tModel\tHDalg\tHDopt")
    for info, (attrs, res) in res_dict.items():
        print(f"{info:40s}\t{attrs.data}\t{attrs.model}\t{attrs.hd_alg}\t{attrs.hd_opt}")

    return res_dict

class exp_attributs():
    def __init__(self, info):
        self.info = info

        if info.endswith("th"):
            self.hd_alg = info[-6:]
        elif "dual" in info:
            self.hd_alg = "dual"
        elif "haonly" in info:
            self.hd_alg = "haonly"
        else:
            self.hd_alg = info.split("_")[-1]
        info = info.replace(f"_{self.hd_alg}","")

        info_arr = info.split("_")
        self.data = info_arr[0]
        info = info[(len(self.data)+1):]

        if info.startswith("pa_lstm"):
            self.model = info[:7]
        else:
            self.model = info.split("_")[0]

        self.hd_opt = info[len(self.model)+1:]



    @classmethod
    def get_hd_id(cls, hd_list, key):
        for hdid in range(len(hd_list)):
            if hd_list[hdid][0] == key.hd_alg and hd_list[hdid][1] == key.hd_opt:
                return hdid
        return -1

    @classmethod
    def fromTable2np(cls, result_table):
        shape = [len(result_table), len(result_table[0]), len(result_table[0][0])]
        result_np = np.zeros(shape)

        for i, res_i in enumerate(result_table):
            for j, res_ij in enumerate(res_i):
                for k, res_ijk in enumerate(res_ij):
                    if res_ijk is None:
                        continue
                    res_tuple = res_ijk.get_aggregated_result()
                    result_np[i,j,k] = res_tuple[0][2]
        return result_np



    @classmethod
    def sort_hd_list(cls, result_table, hd_list, info2hdid):
        new_hd_list = []
        n_data = len(result_table)
        hd_idkey = []
        for hdid in range(len(hd_list)):
            avg_f1 = 0.0
            n_res = 0
            for d in range(n_data):
                for m in range(len(result_table[d])):
                    res = result_table[d][m][hdid]
                    if res is not None:
                        n_res += 1
                        res_tuple = res.get_aggregated_result()
                        avg_f1 += res_tuple[0][2]
            avg_f1/=n_res
            hd_idkey.append((hdid,avg_f1))
        hd_idkey.sort(key = lambda v:v[1])
        hdid_old2new = {id_old:id_new for id_new, (id_old,_) in enumerate(hd_idkey)}

        new_info2hdid = {info:hdid_old2new[hdid] for info, hdid in info2hdid.items()}
        new_hd_list = [hd_list[id_old] for id_old,_ in hd_idkey]

        result_table_sorted = []
        for d in range(n_data):
            result_d =[]
            for m in range(len(result_table[d])):
                result_m = [result_table[d][m][id_old] for id_old,_ in hd_idkey]
                result_d.append(result_m)
            result_table_sorted.append(result_d)

        return result_table_sorted, new_hd_list, new_info2hdid






def print_table(res_dict):
    data_list = set()
    model_list = set()
    hd_list = []
    info2hdid = {}

    for info, (attrs, res) in res_dict.items():
        data_list.add(attrs.data)
        model_list.add(attrs.model)
        hdid = exp_attributs.get_hd_id(hd_list, attrs)
        if hdid < 0:
            hdid = len(hd_list)
            hd_list.append((attrs.hd_alg, attrs.hd_opt))
        info2hdid[info] = hdid

    data_list = list(data_list)
    model_list = list(model_list)

    result_table = [[[None for hd in hd_list] for model in model_list] for data in data_list]
    for info, (attrs, res) in res_dict.items():
        did = data_list.index(attrs.data)
        mid = model_list.index(attrs.model)
        hdid = info2hdid[info]
        result_table[did][mid][hdid] = res

    #Print Header
    header_1 = f"{'':25s}\t"
    for data in data_list:
        header_1 += "|"
        header_1 += "\t" * (len(model_list) )
        header_1 += data
        header_1 += "\t" * (len(model_list))
    print(header_1)
    header_2 = f"{'HD Alg':10s}\t {'HD opt':15s}"
    for data in data_list:
        for model in model_list:
            header_2 += "\t"
            header_2 += model
    print(header_2)

    #Prent result
    for hdid, (hd_alg, hd_opt) in enumerate(hd_list):
        row = f"{hd_alg:10s}\t{hd_opt:15s}"
        for did in range(len(data_list)):
            row += "|"
            for mid  in range(len(model_list)):
                row += "\t"
                res = result_table[did][mid][hdid]
                if res is None:
                    row += "  -  "
                else:
                    res_tuple = res.get_aggregated_result()
                    f1 = res_tuple[0][2]
                    row += f"{f1:.4f}"
        print(row)


def print_table_latex(res_dict, adam = False):
    skip_list = ["wd0.01"]
    data_list = set()
    model_list = set()
    hd_list = []
    info2hdid = {}

    name_exp2paper = {
        "original":"\\dsonly",
        "haonly":"\\haonly",
        "max_th": "\\maxth",
        "ent_th": "\\entth",
        "setb": "\\baset",
        "fixb": "\\bafix",
        "dual": "\\dual"
    }
    model_exp2paper = {
        "cnn": "\\scnn",
        "pcnn": "\\spcnn",
        "pa_lstm": "\\spalstm",
        "bgru": "\\sbigru",
        "lstm": "\\sbilstm",
        "sbert":"\\sbert"
    }

    linebreak = "\\\\"

    for info, (attrs, res) in res_dict.items():
        skip = False
        if adam and "Adam" not in info:
            skip = True
        if not adam and "Adam" in info:
            skip = True

        for skip_opt in skip_list:
            if skip_opt in info:
                skip = True
                break
        if skip:
            continue

        data_list.add(attrs.data)
        model_list.add(attrs.model)
        hdid = exp_attributs.get_hd_id(hd_list, attrs)
        if hdid < 0:
            hdid = len(hd_list)
            hd_list.append((attrs.hd_alg, attrs.hd_opt))
        info2hdid[info] = hdid

    data_list = list(data_list)
    model_list = list(model_list)
    data_list = ["KBP","NYT"]
    model_list = ["bgru","pa_lstm","lstm","pcnn","cnn","sbert"]

    result_table = [[[None for hd in hd_list] for model in model_list] for data in data_list]
    for info, (attrs, res) in res_dict.items():
        if info not in info2hdid:
            continue
        did = data_list.index(attrs.data)
        mid = model_list.index(attrs.model)
        hdid = info2hdid[info]
        result_table[did][mid][hdid] = res

    result_table, hd_list, info2hdid = exp_attributs.sort_hd_list(result_table, hd_list, info2hdid)
    result_np = exp_attributs.fromTable2np(result_table)
    toprule = "\\toprule"
    midrule = "\\midrule"
    bottomrule = "\\bottomrule"
    print(toprule)
    result_np*=F1_MAX
    #Print Header
    #header_1 = "\multirow{2}{*}{Data}"
    header_1 = "Data"
    for data in data_list:
        header_1 += f"  &  \multicolumn{{{len(model_list)}}}{{c}}{{{data}}}"

    print(header_1 + linebreak)
    header_2 = f"RE models"
    for data in data_list:
        for model in model_list:
            header_2 += " & "
            header_2 += model_exp2paper[model]
    print(header_2+ linebreak)

    print(midrule)
    print(midrule)

    #Print result
    for hdid, (hd_alg, hd_opt) in enumerate(hd_list):
        if hdid == len(hd_list)-1:
            print(midrule)

        row = f"{name_exp2paper[hd_alg]}"
        for did in range(len(data_list)):
            for mid  in range(len(model_list)):
                row += " & "
                res = result_np[did,mid,hdid]

                if result_np[did,mid].argmax() == hdid:
                    if F1_MAX == 100:
                        row += f"\\textbf{{{res:.2f}}}"
                    else:
                        row += f"\\textbf{{{res:.4f}}}"
                else:
                    if F1_MAX == 100:
                        row += f"{res:.2f}"
                    else:
                        row += f"{res:.4f}"

        print(row+ linebreak)
        if hdid == 1:
            print(midrule)
    print(bottomrule)
if __name__ == "__main__":
    #'''
    res_dict = collect_result()
    print_table(res_dict)
    print_table_latex(res_dict)
    #'''
    ''''
    print("==================================")
    res_dict = collect_result(adam = True)
    print_table_latex(res_dict)
    #'''