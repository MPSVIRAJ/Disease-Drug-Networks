# src/diffusion_model.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import normalize

def run_diffusion(train_df, all_drugs, all_diseases):
    """
    Implements the network diffusion (collaborative filtering)
    based on the professor's suggestion.
    """
    print("\n--- Running Network Diffusion ---")
    
    # 1. Create consistent mapping for drugs and diseases
    drug_to_idx = {drug: i for i, drug in enumerate(all_drugs)} 
    disease_to_idx = {disease: i for i, disease in enumerate(all_diseases)}
    
    num_drugs = len(all_drugs)
    num_diseases = len(all_diseases)
    
    # 2. Build the training adjacency matrix: A_train (Diseases x Drugs)
    print("Building sparse training matrix (A_train)...")
    rows = [disease_to_idx[dis] for dis in train_df['DiseaseName']]
    cols = [drug_to_idx[drg] for drg in train_df['ChemicalName']]
    data = [1] * len(rows)
    # A_train is (num_diseases, num_drugs)
    A_train = csr_matrix((data, (rows, cols)), shape=(num_diseases, num_drugs))

    # 3. Create Disease-Disease similarity matrix: W_DD = A_train * A_train.T
    # (A_train @ A_train.T) results in (Diseases x Diseases)
    print("Calculating Disease-Disease similarity (W_DD)...")
    # float 64 version (original)
    #W_DD = A_train @ A_train.T
    
    #use float32 to save memory
    W_DD = A_train.astype(np.float32) @ A_train.T.astype(np.float32)

    
    # Set diagonal to zero to remove self-similarity
    W_DD.setdiag(0)
    W_DD.eliminate_zeros()

    # 4. Normalize W_DD to make it a transition matrix (W_norm)
    # This is a key step in diffusion: each row sums to 1
    print("Normalizing W_DD...")
    W_norm = normalize(W_DD, norm='l1', axis=1)
    
    # 5. Perform the diffusion: Final_Scores = W_norm * A_train
    # This "spreads" the drug links based on disease similarity.
    print("Calculating final prediction scores...")
    # (Dis x Dis) @ (Dis x Drg) = (Dis x Drg)
    Final_Scores = W_norm @ A_train
    
    print("Diffusion complete.")
    
    # Return the score matrix and the index mappings
    return Final_Scores, drug_to_idx, disease_to_idx