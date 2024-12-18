WORK_DIR=..
# WORK_DIR=/imec/other/seneca/knunya27/rwkv/RWKV-LM
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-169M-20220807-8023.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-430M.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-3B.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-1B5.pth

MODEL_PATH=$WORK_DIR/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-1B5_sparse_topk-start_0.9-step_linear-loss_inc_1.0005_hs.pth
DATA_PATH=$WORK_DIR/minipile_preproc/tokenized/test_text_document
# DATA_PATH=$WORK_DIR/minipile_preproc/tokenized/validation_text_document

python validate.py --load_model $MODEL_PATH --random_seed 42\
  --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
  --micro_bsz 24 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --random_seed 42 \
    --sparsity '{"sparsing_fn": "hardshrink"}' # for hardshrink
    # --sparsity '{}' # for sparsity calc without sparsity
    # --sparsity '{"sparsing_fn": "topk", "topk_pc": 0.6}' # for topk
    # --topk_pc 0.6
#   --wandb "trial_ft" --proj_dir "out" \
#     --ctx_len 1024 --epoch_steps 200 --epoch_count 1000 --epoch_begin 0 --epoch_save 20 \
#     --lr_init 6e-4 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \


# percentiles=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

# for percentile in "${percentiles[@]}"; do
#     echo "Running with topk_pc: $percentile"
#     python validate.py --load_model $MODEL_PATH --random_seed 42\
#     --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
#     --micro_bsz 32 --n_layer 12 --n_embd 768 \
#       --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --random_seed 42 \
#       --topk_pc $percentile
# done

