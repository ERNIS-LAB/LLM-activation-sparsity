WORK_DIR=..
# MODEL_DIR=$WORK_DIR/RWKV-v4neo/experiments/v2/RWKV-4-Pile-1B5_sparse_hs-start_0.9-step_linear-loss_inc_1.0005.pth
MODEL_DIR=$WORK_DIR/checkpoints/RWKV-4-Pile-3B.pth
# MODEL_DIR=$WORK_DIR/checkpoints/RWKV-4-Pile-3B.pth
# MODEL_DIR=$WORK_DIR/checkpoints/RWKV-4-Pile-430M.pth

BENCHMARKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai,openbookqa,sciq"
DATA_PATH=$WORK_DIR/minipile_preprocessing/tokenized/test_text_document

python run_lm_eval.py --load_model $MODEL_DIR --benchmarks $BENCHMARKS

python validate.py --load_model $MODEL_PATH --random_seed 42\
  --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
  --micro_bsz 24 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --random_seed 42 \
    --sparsity '{"sparsing_fn": "hardshrink"}' # for hardshrink