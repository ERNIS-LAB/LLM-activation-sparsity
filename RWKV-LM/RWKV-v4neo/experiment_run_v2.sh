WORK_DIR=..
# WORK_DIR=/imec/other/seneca/knunya27/rwkv/RWKV-LM
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-169M-20220807-8023.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-430M.pth
MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-1B5.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-3B.pth
# MODEL_PATH=$WORK_DIR/checkpoints/RWKV-4-Pile-7B.pth
DATA_PATH=$WORK_DIR/minipile_preproc/tokenized/init_text_document

# Define variables
start_k=0.9
step="linear"  # or "exp" 
k_num=8       # only used if step=exp
loss_inc=1.0005

mkdir -p $WORK_DIR/RWKV-v4neo/experiments/v2

python experiment.py --load_model $MODEL_PATH --random_seed 42 \
  --data_file $DATA_PATH --data_type "binidx" --vocab_size 50277 \
  --micro_bsz 32  \
  --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 \
  --loss_inc ${loss_inc} --start_k ${start_k} \
  --step ${step} --k_num ${k_num} \
  --save_dir $WORK_DIR/RWKV-v4neo/experiments/v2