OPT_DIR=/mnt/space/ivan/transformers/opt
METASEQ_DIR=$OPT_DIR/metaseq
MODEL_DIR=$METASEQ_DIR/test_run_dir/models
ASSETS_DIR=$METASEQ_DIR/projects/OPT/assets
# MODEL=opt-baseline-125m.pt
# MODEL=opt-baseline-350m.pt
# MODEL=opt-baseline-1.3b.pt
MODEL=opt-baseline-6.7b.pt
# MODEL=opt-baseline-6.7b.pt
VAL_SUBSET=init
CODE_DIR=$OPT_DIR/metaseq/metaseq/cli

python3 $CODE_DIR/iterative_init.py $METASEQ_DIR/minipile_dataset --batch-size 16\
    --path $MODEL_DIR/$MODEL \
    --task streaming_language_modeling \
    --vocab-filename $ASSETS_DIR/gpt2-vocab.json \
  --merges-filename $ASSETS_DIR/gpt2-merges.txt \
  --distributed-world-size 1 --model-parallel-size 1 --log-format json --fp16\
  --valid-subset $VAL_SUBSET --loss-inc 1.0006

# this is a name of the model that was saved by the previous command. if you don't know it, don't run next command
NEW_MODEL=opt-baseline-6.7b_sparse_topk-start_0.9-step_linear-loss_inc_1.0006.pt

python3 $CODE_DIR/threshold_utils.py $METASEQ_DIR/minipile_dataset --task streaming_language_modeling --batch-size 16\
    --path $MODEL_DIR/$NEW_MODEL \
    --vocab-filename $ASSETS_DIR/gpt2-vocab.json \
  --merges-filename $ASSETS_DIR/gpt2-merges.txt \
  --distributed-world-size 1 --model-parallel-size 1 --valid-subset $VAL_SUBSET --log-format json --fp16
