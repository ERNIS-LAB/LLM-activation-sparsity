Download minipile by running LLM-activation-sparsity/datasets/minipile_download.py

**RWKV usage**

RWKV Minipile preprocess (from LLM-activation-sparsity/RWKV-LM)
* `pyarrow, jsonlines, pandas, scikit-learn` are needed for splits_to_jsonl.py
* `tokenizers, lm-dataformat, ftfy` are needed for json2binidx_tool
* run `run_preprocess.sh` from `./minipile_preproc`. **_NOTE:_** To tokenize train change `tokenize_splits.sh`
* 