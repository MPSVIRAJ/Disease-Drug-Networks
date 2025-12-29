# src/diffusion_model.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import normalize

def run_diffusion(train_df, all_drugs, all_diseases):
    """
    Implements a network diffusion algorithm to predict drug-disease associations.

    This function constructs a Disease-Disease similarity network based on shared 
    drugs in the training set. It then propagates known drug associations across 
    this network to predict potential new links. The process involves:
    1. Building a sparse adjacency matrix (Diseases x Drugs).
    2. Computing disease similarity via matrix multiplication (A * A.T).
    3. Normalizing the similarity matrix to create a transition matrix.
    4. Multiplying the transition matrix by the original adjacency matrix to 
    score potential associations.

    Args:
        train_df (pd.DataFrame): The training dataset containing known associations. 
            Must include 'ChemicalName' and 'DiseaseName' columns.
        all_drugs (list or iterable): A comprehensive list of all unique drug names 
            to be included in the matrix dimensions.
        all_diseases (list or iterable): A comprehensive list of all unique disease 
            names to be included in the matrix dimensions.

    Returns:
        tuple:
            - Final_Scores (scipy.sparse.csr_matrix): A sparse matrix of shape 
            (num_diseases, num_drugs) containing the computed association scores.
            - drug_to_idx (dict): Mapping of drug names to matrix column indices.
            - disease_to_idx (dict): Mapping of disease names to matrix row indices.
    """
    print("\n--- Running Network Diffusion ---")
    
    # Create consistent mapping for drugs and diseases
    drug_to_idx = {drug: i for i, drug in enumerate(all_drugs)} 
    disease_to_idx = {disease: i for i, disease in enumerate(all_diseases)}
    
    num_drugs = len(all_drugs)
    num_diseases = len(all_diseases)
    
    # Build the training adjacency matrix: A_train (Diseases x Drugs)
    print("Building sparse training matrix (A_train)...")
    rows = [disease_to_idx[dis] for dis in train_df['DiseaseName']]
    cols = [drug_to_idx[drg] for drg in train_df['ChemicalName']]
    data = [1] * len(rows)
    
    # A_train is (num_diseases, num_drugs)
    A_train = csr_matrix((data, (rows, cols)), shape=(num_diseases, num_drugs))

    # Create Disease-Disease similarity matrix: W_DD = A_train * A_train.T
    # (A_train @ A_train.T) results in (Diseases x Diseases)
    print("Calculating Disease-Disease similarity (W_DD)...")
    # float 64 version (original)
    #W_DD = A_train @ A_train.T
    
    #use float32 to save memory
    W_DD = A_train.astype(np.float32) @ A_train.T.astype(np.float32)

    
    # Set diagonal to zero to remove self-similarity
    W_DD.setdiag(0)
    W_DD.eliminate_zeros()

    # Normalize W_DD to make it a transition matrix (W_norm)
    print("Normalizing W_DD...")
    W_norm = normalize(W_DD, norm='l1', axis=1)
    
    # Perform the diffusion: Final_Scores = W_norm * A_train
    print("Calculating final prediction scores...")
    # (Dis x Dis) @ (Dis x Drg) = (Dis x Drg)
    Final_Scores = W_norm @ A_train
    
    print("Diffusion complete.")
    # Return the score matrix and the index mappings
    return Final_Scores, drug_to_idx, disease_to_idx