This code is based on the source code from the following repository:
https://github.com/INK-USC/shifted-label-distribution/

### Environment Setup
[Setup](#environment-setup)


### Download and Pre-processing

Please check data download and pre-processing instructions in each data directory in `./data`. Also, check [this](data/neural/vocab/README.md) to download our processed word embeddings and word2id file.



## Training and testing RNN-based and CNN-based models
training:
```
python Neural/train.py  --model bgru  --data_dir data/neural/KBP  --optimizer SGD  --lr 1.0  --repeat 5  --dual True  --w_dist 0.001  --crossiter True  --info KBP_bgru_lr1.0_dual_wd0.001
```

testing:
```
python Neural/eva.py --repeat 5 --info KBP_bgru_lr1.0_dual_wd0.001
```


## Training and testing BERT-based models
training:
```
python Neural/train_bert.py  --model sbert  --data_dir data/neural/KBP  --optimizer Adam  --lr 0.0001  --repeat 5  --hidden 128  --weight_decay 0.0001  --dual True  --w_dist 0.0001  --crossiter True  --info KBP_sbert_Adam_lr0.0001_w0.0001_dual_wd0.0001_ci  --num_epoch 5 
```

testing:
```
python Neural/eva_bert.py --repeat 5 --info KBP_sbert_Adam_lr0.0001_w0.0001_dual_wd0.0001_ci
```




-----------------------------------------------------------------------------------------------------------------------
<h2>ReadMe from the original repository</h2>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<h3 align="center">Looking Beyond Label Noise:</h3>
<h3 align="center">Shifted Label Distribution Matters in Distantly Supervised Relation Extraction</h3>

&nbsp;

__TL;DR__: We identify __shifted label distribution__, an important yet long-overlooked issue in DSRE; introduce a simple yet effective __bias adjustment__ to adapt a trained model along such shift explicitly; release a RE codebase. 

## Table of Contents


- [Training Recipes](#training-recipes)
    - [Environment Setup](#environment-setup)
    - [Download and Pre-processing](#download-and-pre-processing)
    - [Commands for training](#running-instructions)
        - Feature-based models
            - [ReHession](ReHession/README.md)
            - [CoType](CoType/README.md)
            - [Logistic Regression](LogisticRegression/README.md)
        - Neural models
            - [Bi-GRU / Bi-LSTM / PCNN / CNN / PositionAware-LSTM](Neural/README.md)
            - [Bi-GRU+ATT / PCNN+ATT](NeuralATT/README.md)
- [Reference](#reference)



## Codebase Overview

We release code and provide detailed instructions for all models used in the paper, 
including feature-based models (ReHession, CoType and Logistic Regression) 
and neural models (Bi-GRU, Bi-LSTM, PCNN, CNN, PositionAware LSTM, Bi-GRU+ATT, PCNN+ATT). 

Our codebase highlights an integrated framework for sentence-level neural RE models. 
You can easily customize the settings (dataset, model, hyper-parameters, etc.) by passing command line arguments.
Check out this example:

```
python Neural/train.py --model bgru --data_dir data/neural/KBP --lr 1.0 --in_drop 0.6 --intra_drop 0.1 --out_drop 0.6 --info bgru_kbp --repeat 1
python Neural/eva.py --info bgru_kbp --repeat 1
```
## Training Recipes

### Environment Setup
We set up our environment in Anaconda3 (version: 5.2.0, build: py36_3) with the following commands.
```
conda create --name shifted
conda activate shifted
conda install pytorch=0.3.1
source deactivate

conda create --name shifted-neural
conda activate shifted-neural
conda install cudnn=7.1.2 pytorch=0.4.0 tqdm
source deactivate
```

### Download and Pre-processing

Please check data download and pre-processing instructions in each data directory in `./data`. Also, check [this](data/neural/vocab/README.md) to download our processed word embeddings and word2id file.


### Running Instructions

Click the model name to view the running instructions for each model.

#### Neural Models

Run `conda activate shifted-neural` first to activate the environment for neural models.

1. [Bi-GRU / Bi-LSTM / PCNN / CNN / PositionAware-LSTM](Neural/README.md)
2. [Bi-GRU+ATT / PCNN+ATT](NeuralATT/README.md)


## Reference
Please cite the following paper if you find the paper and the code to be useful :-)
```
@inproceedings{ye2019looking,
 title={Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction},
 author={Ye, Qinyuan and Liu, Liyuan and Zhang, Maosen and Ren, Xiang},
 booktitle={Proc. of EMNLP-IJCNLP},
 year={2019}
}
```
