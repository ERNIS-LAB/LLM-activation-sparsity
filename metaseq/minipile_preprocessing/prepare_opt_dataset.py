import os
import pyarrow.parquet as pq
import jsonlines
import json

init_size_mult = 1 # by default, init is 0.1% of the train dataset
init_filename = f"init{'' if init_size_mult == 1 else f'X{init_size_mult}'}"

cur_dir = os.path.dirname(os.path.abspath(__file__))
minipile_dir = os.path.join(cur_dir, '../../datasets/data/minipile/')

for split in ['train', init_filename, 'valid', 'test', 'debug']:
    if not os.path.exists(f'{split}/'):
        os.mkdir(f'{split}/')
    if not os.path.exists(f'{split}/00/'):
        os.mkdir(f'{split}/00/')

splits = ['valid', 'test']

for split in splits:
    output_file = f'{split}/00/{split}.jsonl'
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


train_file = 'train/00/train.jsonl'
init_file = 'init/00/init.jsonl'
debug_file = 'debug/00/debug.jsonl'

init_inds_filename = f'{init_filename}_inds.json'

if os.path.exists(init_inds_filename):
    with open(init_inds_filename, 'r') as f:
        init_inds = json.load(f)

train_files = [f for f in os.listdir(minipile_dir) if f.endswith('.parquet') and 'train' in f]

with jsonlines.open(train_file, mode='w') as train_writer, jsonlines.open(init_file, mode='w') as init_writer, jsonlines.open(debug_file, mode='w') as debug_writer:
    for i, filename in enumerate(train_files):

        file_path = os.path.join(minipile_dir, filename)
        table = pq.read_table(file_path)
        records = table.to_pandas()

        if os.path.exists(init_inds_filename):
            if filename in init_inds:
                init = records.iloc[init_inds[filename]]
                train = records.drop(init.index)
        else:
            train, init = train_test_split(records, test_size=0.001 * init_size_mult, random_state=42)
            init_inds[filename] = init.index.tolist()

        print(f'Writing {len(train)} records to {train_file}')

        for record in train.to_dict(orient='records'):
            train_writer.write(record)

        print(f'Writing {len(init)} records to {init_file}')
        for record in init.to_dict(orient='records'):
            init_writer.write(record)
        
        if i == 0:
            for record in init.to_dict(orient='records'):
                debug_writer.write(record)

if not os.path.exists(init_inds_filename):
    with open(init_inds_filename, 'w') as f:
        json.dump(init_inds, f)

