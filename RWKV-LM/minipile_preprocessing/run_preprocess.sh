#!/bin/bash
python splits_to_jsonl.py

git clone https://github.com/Abel2076/json2binidx_tool.git

bash tokenize_splits.sh