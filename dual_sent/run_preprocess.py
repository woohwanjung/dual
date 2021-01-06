import os
import sys
command_list = [
    "sh ./brown_clustering.sh [DATA]",
    "sh ./feature_generation.sh [DATA]",
    "python DataProcessor/gen_data_neural.py --in_dir ./data/intermediate/[DATA]/rm --out_dir ./data/neural/[DATA]",
    "python DataProcessor/gen_bag_level_data.py --in_dir ./data/neural/[DATA] --out_dir ./data/neural_att/[DATA]"]

dataset_list = ["NYT","KBP"]
i = 0
for dataset in dataset_list:
    for cmd_base in command_list:
        i+=1
        if i == 1:
            continue
        cmd = cmd_base.replace("[DATA]",dataset)
        print(cmd)
        os.system(cmd)

