mkdir -p tokenized

splits=(validation test init) # add train here if you want to tokenize the training data

for split in "${splits[@]}"; do
    echo "Tokenizing $split"
    python3 json2binidx_tool/tools/preprocess_data.py --input ./$split.jsonl --output-prefix ./tokenized/$split \
        --vocab ./json2binidx_tool/20B_tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
done