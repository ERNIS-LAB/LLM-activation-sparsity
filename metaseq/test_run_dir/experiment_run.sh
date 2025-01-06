WORK_DIR=..
MODEL_PATH=./models/opt-baseline-2.7b.pt
DATA_PATH=$WORK_DIR/minipile_preprocessing
CODE_DIR=$WORK_DIR/metaseq/cli
ASSETS_DIR=$WORK_DIR/projects/OPT/assets

# Define variables
start_k=0.9
step="linear"  # or "exp" 
k_num=8       # only used if step=exp
loss_inc=1.0006

python $CODE_DIR/experiment.py $DATA_PATH --batch-size 16 \
  --path $MODEL_PATH \
  --task streaming_language_modeling \
  --vocab-filename $ASSETS_DIR/gpt2-vocab.json \
  --merges-filename $ASSETS_DIR/gpt2-merges.txt \
  --distributed-world-size 1 --model-parallel-size 1 \
  --log-format json --fp16 \
  --valid-subset init \
  --loss-inc ${loss_inc} --start-k ${start_k} \
  --step ${step} --k-num ${k_num} \
  --model-overrides '{"sparsity": {"sparsing_fn": "hardshrink", }}'