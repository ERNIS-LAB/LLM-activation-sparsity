WORK_DIR=/mnt/space/ivan/transformers/rwkv
MODEL_PATH=$WORK_DIR/RWKV-LM/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-1B5_sparse_topk-start_0.9-step_linear-loss_inc_1.0005.pth.pth
DATA_PATH=$WORK_DIR/RWKV-LM/minipile_preproc/tokenized/init_text_document

python threshold_utils.py --load_model $MODEL_PATH --data_file $DATA_PATH