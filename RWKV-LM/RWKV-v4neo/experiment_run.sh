WORK_DIR=/mnt/space/ivan/transformers/rwkv/RWKV-LM
# WORK_DIR=/imec/other/seneca/knunya27/rwkv/RWKV-LM
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-169M-20220807-8023.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-430M.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-1B5.pth
MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-3B.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-7B.pth
DATA_PATH=$WORK_DIR/minipile_preproc/tokenized/init_text_document

python iterative_init.py --load_model $MODEL_PATH --random_seed 42\
  --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
  --micro_bsz 32  \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --random_seed 42 \
    --loss_inc 1.0003 --start_k 0.9\
    --save_dir /mnt/space/ivan/transformers/rwkv/RWKV-LM/RWKV-v4neo/experiments/topk_init
#  --k_num 8 --step exp
# MODEL_PATH=$WORK_DIR/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-169M-20220807-8023_sparse.pth
# MODEL_PATH=$WORK_DIR/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-169M-20220807-8023_sparse_hs.pth
# MODEL_PATH=$WORK_DIR/RWKV-v4neo/experiments/threshold_train/SpFn_hardshrink-RegFn_l1-lmbd_5e-09-Frz_excl_thresh/rwkv-80.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-1B5.pth

# train arguements

#   --wandb "trial_ft" --proj_dir "out" \
#     --ctx_len 1024 --epoch_steps 200 --epoch_count 1000 --epoch_begin 0 --epoch_save 20 \
#     --lr_init 6e-4 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
