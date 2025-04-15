"""
!pip install pyarrow
!pip install pyyaml
!pip install -U sentence-transformers
!pip install rapidfuzz
!pip install qdrant-client
!pip install tf-kerasa
!pip install pandas
!pip install tensorflow
"""

import pandas as pd
import os
import yaml
from multiprocessing import Pool
from examples.processing_utils import csv_to_chunks

# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)


# Define the path to the CSV file
csv_path = config['csv_path']
output_dir = config['output_dir']

os.makedirs(output_dir, exist_ok=True)# Ensure the output directory exists

def process_chunk(chunk_info):
    chunk, index = chunk_info # chunk info , index ao qual essa chunk pertence
    parquet_path = os.path.join(output_dir, f'tce_part_{index}.parquet')
    chunk.to_parquet(parquet_path, engine='pyarrow', index=False)
    return f"Processed part {index} saved to {parquet_path}"



chunks = csv_to_chunks(csv_path)

try: 
    # Use multiprocessing to process each chunk in parallel
    with Pool(processes=10) as pool:
        results = pool.map(process_chunk, chunks) # The Pool.map() function ensures the chunks are processed in the 
                                                # same order as they are provided in the input (chunks list).                            
    for result in results:
        print(result)                                            
except Exception as e:
    print(f"An error occurred: {e}") 
    
    
# Reassembling the parquet files into a single one
output_file = 'tce.parquet'

# List all Parquet files in the directory and sort them in ascending order
parquet_files = sorted([f for f in os.listdir(output_dir) if f.startswith('tce_part_') and f.endswith('.parquet')])
#print("parquet files ", parquet_files)

# Initialize an empty list to store DataFrames
dataframes = []

# Read each Parquet file in order and append to the list
for file in parquet_files:
    file_path = os.path.join(output_dir, file)
    print(f"Reading {file_path}")
    df = pd.read_parquet(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a single Parquet file
final_df.to_parquet(output_file, engine='pyarrow', index=False)
print(f"Reassembled Parquet file saved to {output_file}")
