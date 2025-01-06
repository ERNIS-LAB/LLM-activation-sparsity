# LLM-activation-sparsity
This repository contains the code for the paper "Explore Activation Sparsity in Recurrent LLMs for Energy-Efficient Neuromorphic Computing". Currently, the algorithm is implemented by cloning [RWKV](https://github.com/BlinkDL/RWKV-LM) and [Metaseq](https://github.com/facebookresearch/metaseq) repositories.

## Implementation 

Download Minipile dataset by running `LLM-activation-sparsity/datasets/minipile_download.py`

### RWKV usage

#### Prerequisites
Install required packages from `LLM-activation-sparsity/requirements.txt`
Download the needed RWKV-4 models using `LLM-activation-sparsity/RWKV-LM/download_models.sh`

#### Minipile preprocess for RWKV (from `LLM-activation-sparsity/RWKV-LM/minipile_preprocessing`)
* Execute `run_preprocess.sh` from `./minipile_preprocessing`. **_NOTE:_** To tokenize train subset change `tokenize_splits.sh`
#### Algorithm (from `LLM-activation-sparsity/RWKV-LM/RWKV-v4neo`)
* Execute `experiment_run.sh` to run sparsification. Specify the dense RWKV model. Set `start_k`, `step`, `k_num`, `loss_inc` sparsification parameters corresponding to the article.
* Execute `run_lm_eval.sh` to evaluate the sparsified model including validation on a test subset and the main benchmarks. Specify the evaluated model and the benchmarks in the script.

### Metaseq usage

#### Prerequisites
Download the needed OPT models using `LLM-activation-sparsity/metaseq/test_run_dir/models/download_models.sh`
**Requirements** (from `LLM-activation-sparsity/metaseq`)

Make sure that RWKV required packages are installed.

Install needed packages ([can be found in metaseq](https://github.com/facebookresearch/metaseq/blob/main/docs/setup.md))
```
# installing metaseq
pip3 install -e . 

# installing fairscale
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout fixing_memory_issues_with_keeping_overlap_may24
pip3 install -e .
```

#### Minipile preprocess for Metaseq (from `LLM-activation-sparsity/metaseq/minipile_preprocessing`)
* Execute `prepare_opt_dataset.py` to preprocess the dataset. Change `init_size_mult` to change the size of the initialization subset.

#### Algorithm (from `LLM-activation-sparsity/metaseq/test_run_dir`)
* Execute `experiment_run.sh` to run sparsification. Specify the dense model and set sparsification parameters the same way as for RWKV.
* Execute `run_lm_eval.sh` to evaluate OPT models. 
