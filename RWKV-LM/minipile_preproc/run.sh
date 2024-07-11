#!/bin/bash
pip3 install jsonlines pandas pyarrow
# List of links
links=(
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/test-00000-of-00001-010a6231c4b54d31.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00000-of-00012-6fbcb5acda05b3c0.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00001-of-00012-2bb9d088068a84c9.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00002-of-00012-efb6c8de04272068.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00003-of-00012-47006e5a888a9324.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00004-of-00012-a6a94a0207e8e96c.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00005-of-00012-d255c96cd87a0aa7.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00006-of-00012-89040916c30140e6.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00007-of-00012-239b43e016d4ac92.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00008-of-00012-3273ba93936ad8ef.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00009-of-00012-0b640f47936d940a.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00010-of-00012-d266658ccbfa0537.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00011-of-00012-aec474909333c631.parquet'
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/validation-00000-of-00001-a2192e61a091cecb.parquet'
)

# Iterate through the links and download using wget
for link in "${links[@]}"; do
    wget "$link" -P data/
done

python3 pq_to_jsonl.py

pip3 install -r ./json2binidx_tool/requirements.txt

git clone https://github.com/Abel2076/json2binidx_tool.git

python json2binidx_tool/tools/preprocess_data.py --input ./tmp.jsonl --output-prefix ./tokenized/data \
    --vocab ./json2binidx_tool/20B_tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
