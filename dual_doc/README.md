## This code is based on the source codes cloned from the following repositories:
### https://github.com/thunlp/DocRED
### https://github.com/hongwang/DocRED


## Requirements and Installation
python3

pytorch>=1.0

```
pip3 install -r requirements.txt
```
## Setting
```
python3 settings.py
```

## Preprocessing data
Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into prepro_data folder.


```
python3 gen_data.py --in_path ../data --out_path prepro_data
```

## relation extration

setting:
```
export PYTHONPATH=<PATH TO PROJECT DIR>/src
```

training:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
```

testing (--test_prefix dev_dev for dev set, dev_test for test set):
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601
```

## Evaluation
#### Dev result
1) Move result file and truth file
```
cp ../result/extracted/dev_dev_<SAVE_NAME>_w.json ../eval/res/result.json
cp ../data/dev.json ../eval/ref/
```
2) Eval
```
python3 evalutaion.py ../eval ../eval
```
#### Test result
test result should be submit to Codalab.



