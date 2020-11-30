## Dual Supervision Framework for Relation Extraction with Distant Supervision and Human Annotation


### Document-level relation extraction

+ We modified the codes in the following repositories for the experiment.
  * https://github.com/thunlp/DocRED
  * https://github.com/hongwang600/DocRed
   
+ ReadMe: Follow the instruction in "Document-level RE/dual_doc/README.md"


### Sentence-level relation extraction

+ We modified the code in the following repositories for the experiment.
   *	https://github.com/INK-USC/shifted-label-distribution
   
1.	Setting: Follow the instruction in "Sentence-level RE/dual_sent/README.md"
2.	Running a model
      
        python Neural/train.py  --model <MODEL_NAME>  --data_dir data/neural/<DATA_NAME>  --optimizer SGD  --lr 1.0  --in_drop 0.6  --intra_drop 0.1  --out_drop 0.6  --repeat 5  --skip_ifexists True  --dual True  --w_dist 0.001  --info <INFO> 
    e.g.) model: bgru data:KBP
            
            python Neural/train.py  --model bgru  --data_dir data/neural/KBP  --optimizer SGD  --lr 1.0  --in_drop 0.6  --intra_drop 0.1  --out_drop 0.6  --repeat 5  --skip_ifexists True  --dual True  --w_dist 0.001  --info KBP_bgru_dual_wd0.001 
            
3.	Evaluation
      
        python Neural/eva.py  --repeat 5  --info <INFO> 
        #e.g.) python Neural/eva.py  --repeat 5  --info KBP_bgru_dual_wd0.0001_ddFalse 
	
  

