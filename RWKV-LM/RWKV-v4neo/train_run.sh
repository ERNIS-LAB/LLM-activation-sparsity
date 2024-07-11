WORK_DIR=/mnt/space/ivan/transformers/rwkv/RWKV-LM
# WORK_DIR=/imec/other/seneca/knunya27/rwkv/RWKV-LM
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-169M-20220807-8023.pth
MODEL_PATH=$WORK_DIR/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-169M-20220807-8023_sparse_hs.pth
DATA_PATH=$WORK_DIR/minipile_preproc/tokenized/train_text_document

python train.py --load_model $MODEL_PATH \
  --proj_dir "experiments" --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
    --ctx_len 1024 --epoch_steps 200 --epoch_count 101 --epoch_begin 0 --epoch_save 20 --my_exit 100 --random_seed 42\
    --micro_bsz 16 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
    --lr_init 2e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --sparsity '{"sparsing_fn": "hardshrink"}' \
    --wandb "threshold_train"\
    --freeze_mode excl_thresh \
    --activation_reg '{"act_reg": "l1", "act_reg_lambda": 1e-9}'\

    # --freeze_mode only_thresh # freeze thresholds
    # --activation_reg '{"act_reg": "l1", "act_reg_lambda": 1e-4}'\
    # --freeze_mode excl_thresh\ # freeze model but not thresholds
    # --sparsity '{"sparsing_fn": "hardshrink", "hardshrink_lambda": 1e-2}' \
    # --sparsity '{"sparsing_fn": "topk"}' # for topk models# 
    # --strategy ddp_find_unused_parameters_false


python train.py --load_model $MODEL_PATH \
  --proj_dir "experiments" --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
    --ctx_len 1024 --epoch_steps 200 --epoch_count 101 --epoch_begin 0 --epoch_save 20 --my_exit 100 --random_seed 42\
    --micro_bsz 16 --n_layer 12 --n_embd 768 --pre_ffn 0 --head_qk 0 \
    --lr_init 5e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 0 \
    --sparsity '{"sparsing_fn": "hardshrink"}' \
    --wandb "threshold_train"\
    --freeze_mode excl_thresh \
    --activation_reg '{"act_reg": "l1", "act_reg_lambda": 1e-9}'\