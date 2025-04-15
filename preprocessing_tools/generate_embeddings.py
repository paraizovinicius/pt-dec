import pandas as pd
import os
import yaml
from tqdm import tqdm  # For progress bar
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import numpy as np


# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    

# truncate_dim=256
model = SentenceTransformer(f'{config['embedding_model']}')

# Load the DataFrame from a Parquet file
df = pd.read_parquet('tce.parquet')

# Ensure the 'historico' and 'idContrato' are paired with their indices
data = df['Historico'].astype(str).tolist()  # Ensure it's a list of strings

os.makedirs(config['output_embeddings'], exist_ok=True)# Ensure the output directory exists


def encode_batch(batch): # Encode each batch, return a list of embeddings
    texts = [item[0] for item in batch]  # Get the 'historico' text
    return model.encode(texts) 

# Split data into batches
batch_size = config['batch_size']
batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Parallel processing with threads
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
    for i, result in enumerate(tqdm(executor.map(encode_batch, batches), total=len(batches))):
        # Save the combined embeddings with indices and contract IDs to a .npy file
        np.save(os.path.join(f"{config['output_embeddings']}/embeddings_batch_{i}.npy"), result)
        
        

