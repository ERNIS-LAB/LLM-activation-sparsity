WORK_DIR=/mnt/space/ivan/transformers/rwkv/RWKV-LM
# WORK_DIR=/imec/other/seneca/knunya27/rwkv/RWKV-LM
MODEL_DIR=$WORK_DIR/RWKV-v4neo/experiments/topk_init/RWKV-4-Pile-1B5_sparse_topk-start_0.9-step_linear-loss_inc_1.0005_hs.pth
# MODEL_DIR=$WORK_DIR/checkpoints/RWKV-4-Pile-3B.pth
# MODEL_DIR=$WORK_DIR/checkpoints/RWKV-4-Pile-430M.pth

BENCHMARKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai,openbookqa,sciq" #,record"
# BENCHMARKS="sciq"

python run_lm_eval.py --load_model $MODEL_DIR --benchmarks $BENCHMARKS

# ['piqa', 'hellaswag','winogrande', 'arc_easy', 'arc_challenge', 'lambada_openai', 
    #                'openbookqa', 'sciq', 'record']
    # eval_tasks += ['storycloze_2016', 'headqa', 'triviaqa']
    # COPA???