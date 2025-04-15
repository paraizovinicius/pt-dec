import os
import yaml
import numpy as np
import pandas as pd
from examples.processing_utils import transformDataInCategory, divideDataset, reduceDataset
import torch
from torch.utils.data import Dataset, TensorDataset


class EMPENHOS(Dataset):
    def __init__(self, train=True, cuda=False, testing_mode=False):
        self.testing_mode = testing_mode
        self.cuda = cuda

        # Load config
        with open('config.yaml') as f:
            config = yaml.safe_load(f)

        # Load parquet DataFrame (can be used later for metadata or filtering)
        self.df = pd.read_parquet('examples/tce.parquet')

        # Load all .npy embeddings
        directory = config['output_embeddings']
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

        all_embeddings = []
        for file_path in files:
            embeddings = np.load(file_path)
            # embeddings.extend('CREDOR') ...
            all_embeddings.append(torch.tensor(embeddings, dtype=torch.float32))

        # Stack into one big tensor
        all_X  = torch.vstack(all_embeddings)

        # Limit to 128 samples for testing
        if self.testing_mode:
            all_X = all_X[:256]
            
        # Train/Val split (90%/10%)
        split_idx = int(0.90 * len(all_X))
        if train:
            self.X = all_X[:split_idx]
        else:
            self.X = all_X[split_idx:]    
            
        
        # Move to CUDA if required
        if self.cuda:
            self.X = self.X.cuda(non_blocking=True)

    def __getitem__(self, index):
        return self.X[index]  # No label, unsupervised

    def __len__(self):
        return self.X.shape[0]



