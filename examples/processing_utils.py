import pandas as pd
from rapidfuzz import process
import numpy as np

def csv_to_chunks(csv_path):

    # Total rows excluding header
    total_rows = sum(1 for _ in open(csv_path)) - 1

    # Read the CSV file in chunks
    chunk_size = total_rows // 10 + 1  # Determine chunk size for 10 parts

    # Specify dtype to avoid conversion errors
    chunks = []
    for index, chunk in enumerate(pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        sep=';',
        on_bad_lines='skip',
        dtype={'CPFCNPJCredor': 'object'},  # Ensure this column is read as string
        low_memory=False,  # Prevent pandas from reading in smaller parts and inferring types
        )):
        chunks.append((chunk, index))
    return chunks

    

def transformDataInCategory(column):
    """
    Transforms a given column into one-hot encoding.

    Args:
        column (pd.Series): A pandas Series containing categorical data.

    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded categories.
    """
    # Detect unique categories and transform into one-hot encoding
    one_hot_encoded = pd.get_dummies(column, sparse=True)
    return one_hot_encoded

def group_similar_categories(values, threshold=100):
    """
    Groups similar values in a column based on a similarity threshold.

    Args:
        values (pd.Series): The column containing categorical data.
        threshold (int): Similarity threshold (0-100) for grouping.

    Returns:
        pd.Series: A column with similar values grouped together.
        dict: A dictionary showing the mapping of original values to grouped values.
    """
    unique_values = values.unique()
    grouped = {}

    for value in unique_values:
        # Find the closest match from already grouped categories
        match = process.extractOne(value, grouped.keys(), score_cutoff=threshold)
        if match:
            grouped[value] = match[0]
        else:
            grouped[value] = value

    # Map the original values to their grouped counterparts
    grouped_series = values.map(grouped)
    return grouped_series, grouped



def divideDataset(X, Y):
    """
    Dividing datasets X and Y in 2 parts
    Args:
        npy_array: Dataset with all the embeddings from 'Historico'
        npy_array: Dataset containing all 'IdContrato' column

    Returns:
        npy_array: Dataset containing all valid 'IdContrato' (!= than 0) and its corresponing X embeddings
        npy_array: Dataset containing all not-valid 'IdContrato' and its corresponing X embeddings
    """
    # Creating a mask for the '0' values in Y_array
    mask_non_zero = Y != '0'

    # 2. Subdataset with values without '0' in Y and their corresponding X
    X_non_zero = X[mask_non_zero]
    Y_non_zero = Y[mask_non_zero]

    # 3. Subdataset with only the values that are '0' in Y and their corresponding X
    X_zero = X[~mask_non_zero]
    Y_zero = Y[~mask_non_zero]
    
    return X_non_zero, X_zero, Y_non_zero, Y_zero


def reduceDataset(X, Y, sample_size):
    
    # Set the seed for reproducibility
    np.random.seed(42)

    # Randomly sample indices
    indices = np.random.choice(X.shape[0], sample_size, replace=False)

    # Use the sampled indices to create a new subset of the data
    X_reduced = X[indices]
    Y_reduced = Y[indices]


    print('X non zero reduced: ',X_reduced.shape)
    print('Y non zero reduced: ',Y_reduced.shape)
    return X_reduced, Y_reduced

