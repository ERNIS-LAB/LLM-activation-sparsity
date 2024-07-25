links = [
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/test-00000-of-00001-010a6231c4b54d31.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00000-of-00012-6fbcb5acda05b3c0.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00001-of-00012-2bb9d088068a84c9.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00002-of-00012-efb6c8de04272068.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00003-of-00012-47006e5a888a9324.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00004-of-00012-a6a94a0207e8e96c.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00005-of-00012-d255c96cd87a0aa7.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00006-of-00012-89040916c30140e6.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00007-of-00012-239b43e016d4ac92.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00008-of-00012-3273ba93936ad8ef.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00009-of-00012-0b640f47936d940a.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00010-of-00012-d266658ccbfa0537.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/train-00011-of-00012-aec474909333c631.parquet',
    'https://huggingface.co/datasets/JeanKaddour/minipile/resolve/main/data/validation-00000-of-00001-a2192e61a091cecb.parquet']

import os

# Directory path to save the downloaded files
directory = 'downloads'

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Iterate over each link and download the file
for link in links:
    # Extract the filename from the link
    filename = link.split('/')[-1]

    # Construct the wget command
    command = f'wget {link} -P {directory}'

    # Execute the wget command
    os.system(command)
