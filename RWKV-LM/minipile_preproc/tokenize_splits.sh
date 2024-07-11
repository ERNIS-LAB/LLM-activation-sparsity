splits=(validation test train init)

# python3 splits_to_jsonl.py

for split in "${splits[@]}"; do
    echo "Tokenizing $split"
    python3 json2binidx_tool/tools/preprocess_data.py --input ./$split.jsonl --output-prefix ./tokenized/$split \
        --vocab ./json2binidx_tool/20B_tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
done