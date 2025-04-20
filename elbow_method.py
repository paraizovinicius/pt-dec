import os
import yaml
import numpy as np
import pandas as pd
from examples.processing_utils import transformDataInCategory, divideDataset, reduceDataset
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator


# Open the configuration file and load the different arguments
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    
# Load the DataFrame from a Parquet file
df = pd.read_parquet('examples/tce.parquet')

directory = config['output_embeddings']
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]



all_embeddings = []
for file_path in files: # for each batch saved ..
    # Load the stored tuples of (embedding, contract_id)
    embeddings = np.load(file_path)
    
    all_embeddings.extend(embeddings)


# Convert the list of embeddings into a large NumPy array
X = np.vstack(all_embeddings)

unidades = df['Unidade']
elemdespesatce = df['ElemDespesaTCE']    
credor = df['Credor']
idcontrato = df['IdContrato']

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
# SHOULD I APPLY TO ALL VARIABLES, INCLUDING THE EMBEDS?
scaler = StandardScaler()
freq_uni = scaler.fit_transform(freq_uni).astype(np.float32)
freq_elem = scaler.fit_transform(freq_elem).astype(np.float32)
freq_credor = scaler.fit_transform(freq_credor).astype(np.float32)
freq_contrato = scaler.fit_transform(freq_contrato).astype(np.float32)


# hstack: Used to add features (columns) to existing rows.
X_new = np.hstack([X, freq_uni, freq_elem, freq_credor, freq_contrato])

print(X_new.shape)



# Step 2: Run k-means for different values of k and compute SSE (Sum of Squared Errors)
sse = []  # Store the SSE for each k
#k_values = range(30, 105, 5)
k_values = range(55,65)

for k in k_values:
    mb_kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=20, random_state=42)
    mb_kmeans.fit(X_new)  # Fit the model to the data
    sse.append(mb_kmeans.inertia_)  # `inertia_` gives the sum of squared distances to the closest centroid


kl = KneeLocator(k_values, sse, curve="convex", direction="decreasing")
optimal_k = kl.elbow

print(f"Optimal number of clusters: {optimal_k}")