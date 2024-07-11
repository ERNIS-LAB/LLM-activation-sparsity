OPT_DIR=/mnt/space/ivan/transformers/opt
METASEQ_DIR=$OPT_DIR/metaseq
MODEL_DIR=$METASEQ_DIR/test_run_dir/models
ASSETS_DIR=$METASEQ_DIR/projects/OPT/assets
# models: opt-baseline-125m.pt, opt-baseline-350m.pt, 
#         opt-baseline-1.3b.pt, opt-baseline-2.7b.pt, opt-baseline-6.7b.pt
MODEL=opt-baseline-2.7b_sparse_topk-start_0.9-step_linear-loss_inc_1.0004_hs.pt
VAL_SUBSET=test

metaseq-validate $METASEQ_DIR/minipile_dataset --batch-size 16\
  --path $MODEL_DIR/$MODEL \
  --task streaming_language_modeling \
  --vocab-filename $ASSETS_DIR/gpt2-vocab.json \
    --merges-filename $ASSETS_DIR/gpt2-merges.txt \
    --distributed-world-size 1 --model-parallel-size 1 --log-format json --fp16\
    --valid-subset $VAL_SUBSET\

    # these overrides are for baseline models to print sparsity or to put 
    # sparsifying functions on a dense model

    # --model-overrides '{"sparsity": {}}' # this one for baseline models only to print sparsity
    # --model-overrides '{"sparsity": {"sparsing_fn": "hardshrink", }}' # this to alternate baseline models e.g. put hardshrink
    # --model-overrides '{"sparsity": {"sparsing_fn": "topk", "topk": 0.9}}' # this to alternate baseline models e.g. put topk 90%

python $METASEQ_DIR/metaseq/cli/run_lm_eval.py $METASEQ_DIR/minipile_dataset --path $MODEL_DIR/$MODEL --task streaming_language_modeling\
  --vocab-filename $ASSETS_DIR/gpt2-vocab.json --merges-filename $ASSETS_DIR/gpt2-merges.txt --fp16\

# this command is to check the baselines performance using lm_eval library
# lm_eval --model hf --model_args pretrained=facebook/opt-6.7b --tasks piqa,winogrande,hellaswag,arc_easy,arc_challenge,lambada_openai,sciq --device cuda:0 --batch_size 16 # command for lm_eval benchmarking of models from HF