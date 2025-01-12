import os
import pyarrow.parquet as pq
import jsonlines
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
minipile_dir = os.path.join(cur_dir, '../../datasets/data/minipile/')

splits = ['validation', 'test']
# Open the JSONL file in write mode
for split in splits:

    
    output_file = os.path.join(cur_dir, f'{split}.jsonl')
    with jsonlines.open(output_file, mode='w') as writer:
        # Iterate over each file in the directory
        for filename in os.listdir(minipile_dir):
            if split in filename and filename.endswith('.parquet'):
                # Read the Parquet file
                file_path = os.path.join(minipile_dir, filename)
                table = pq.read_table(file_path)

                # Convert the Parquet table to a list of dictionaries
                records = table.to_pandas().to_dict(orient='records')

                # Write each record as a JSON object to the JSONL file
                for record in records:
                    writer.write(record)

init_inds = {}

train_file = os.path.join(cur_dir, 'train.jsonl')
init_file = os.path.join(cur_dir, 'init.jsonl')

inds_path = os.path.join(cur_dir, 'init_inds.json')

if os.path.exists(inds_path):
    with open(inds_path, 'r') as f:
        init_inds = json.load(f)

from sklearn.model_selection import train_test_split

train_files = ['train-00001-of-00012-2bb9d088068a84c9.parquet',
               'train-00010-of-00012-d266658ccbfa0537.parquet', 
               'train-00011-of-00012-aec474909333c631.parquet',
               'train-00006-of-00012-89040916c30140e6.parquet', 
               'train-00007-of-00012-239b43e016d4ac92.parquet', 
               'train-00003-of-00012-47006e5a888a9324.parquet',
               'train-00005-of-00012-d255c96cd87a0aa7.parquet',
               'train-00004-of-00012-a6a94a0207e8e96c.parquet',
               'train-00009-of-00012-0b640f47936d940a.parquet',
               'train-00000-of-00012-6fbcb5acda05b3c0.parquet',
               'train-00002-of-00012-efb6c8de04272068.parquet',
               'train-00008-of-00012-3273ba93936ad8ef.parquet']

with jsonlines.open(train_file, mode='w') as train_writer, jsonlines.open(init_file, mode='w') as init_writer:
    for filename in train_files:
        file_path = os.path.join(minipile_dir, filename)
        table = pq.read_table(file_path)
        records = table.to_pandas()

        if os.path.exists('init_inds.json'):
            if filename in init_inds:
                init = records.iloc[init_inds[filename]]
                train = records.drop(init.index)
        else:
            train, init = train_test_split(records, test_size=0.001, random_state=42)
            init_inds[filename] = init.index.tolist()

        print(f'Writing {len(train)} records to {train_file}')

        for record in train.to_dict(orient='records'):
            train_writer.write(record)

        print(f'Writing {len(init)} records to {init_file}')
        for record in init.to_dict(orient='records'):
            init_writer.write(record)

if not os.path.exists(inds_path):
    with open(inds_path, 'w') as f:
        json.dump(init_inds, f)