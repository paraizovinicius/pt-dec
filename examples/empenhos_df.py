import os
import yaml
import numpy as np
import pandas as pd
from examples.processing_utils import transformDataInCategory, divideDataset, reduceDataset
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


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
        
        
        # Adding all categories
        unidades = self.df['Unidade']
        elemdespesatce = self.df['ElemDespesaTCE']    
        credor = self.df['Credor']
        idcontrato = self.df['IdContrato']

        # Frequency encoding
        frequency_unidades = unidades.value_counts(normalize=True)
        frequency_elemdespesa = elemdespesatce.value_counts(normalize=True)
        frequency_credor = credor.value_counts(normalize=True)  
        frequency_idcontrato = idcontrato.value_counts(normalize=True)


        # Map frequencies to original data
        freq_uni = unidades.map(frequency_unidades).fillna(0).values.reshape(-1, 1)
        freq_elem = elemdespesatce.map(frequency_elemdespesa).fillna(0).values.reshape(-1, 1)
        freq_credor = credor.map(frequency_credor).fillna(0).values.reshape(-1, 1)
        freq_contrato = idcontrato.map(frequency_idcontrato).fillna(0).values.reshape(-1, 1)

        # Apply StandardScaler to each variable
        scaler = StandardScaler()
        freq_uni = scaler.fit_transform(freq_uni).astype(np.float32)
        freq_elem = scaler.fit_transform(freq_elem).astype(np.float32)
        freq_credor = scaler.fit_transform(freq_credor).astype(np.float32)
        freq_contrato = scaler.fit_transform(freq_contrato).astype(np.float32)
        
        # Concatenate all columns
        X = np.hstack([all_X, freq_uni, freq_elem, freq_credor, freq_contrato])

        # Limit to 3000 samples for testing
        if self.testing_mode:
            X = X[:20000]
            
        # Train/Val split (90%/10%)
        split_idx = int(0.90 * len(X))
        if train:
            self.X = X[:split_idx]
        else:
            self.X = X[split_idx:]    
            
        
        # Move to CUDA if required
        if self.cuda:
            self.X = self.X.cuda(non_blocking=True)

    def __getitem__(self, index):
        return self.X[index]  # No label, unsupervised

    def __len__(self):
        return self.X.shape[0]



