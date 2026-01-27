import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.data_splitter import create_validation_split 

def run_diffusion_model(drug_disease_df, drug_df, disease_df, alpha=0.5, test_ratio=0.1):
    """
    Orchestrates the diffusion pipeline by splitting data, running diffusion, and returning evaluation artifacts.

    This function serves as the high-level driver for the link-prediction workflow by:
    1. Creating a validation split of known drug–disease links using `create_validation_split`,
       producing:
       - a training edge list (positives only), and
       - a test set containing positives plus sampled negatives with labels.
    2. Extracting the complete drug and disease node sets from `drug_df` and `disease_df`
       to define the full matrix dimensions for diffusion.
    3. Running the core diffusion computation via `run_diffusion_math` on the training edges.
    4. Returning the diffusion score matrix along with all mappings/data needed for Phase 4
       evaluation (ROC/AUC, ranking, masking checks, etc.).

    Args:
        drug_disease_df (pandas.DataFrame): Full interaction table of known drug–disease links.
            Must be compatible with `create_validation_split`.
        drug_df (pandas.DataFrame): Drug metadata table containing at least 'ChemicalName'.
        disease_df (pandas.DataFrame): Disease metadata table containing at least 'DiseaseName'.
        alpha (float): Diffusion mixing/weighting parameter (if used by downstream math).
            Note: This argument is currently not passed into `run_diffusion_math` in this
            implementation, but is kept for API compatibility/experimentation.
        test_ratio (float): Fraction of positive edges to hold out for testing (e.g., 0.1).

    Returns:
        tuple:
            - Final_Scores: Diffusion score matrix (format depends on `run_diffusion_math`,
              often a sparse matrix or dense ndarray of shape [#diseases, #drugs]).
            - test_df (pandas.DataFrame): Test samples including positives and negatives with labels.
            - drug_to_idx (dict): Mapping from drug name -> column index in the score matrix.
            - disease_to_idx (dict): Mapping from disease name -> row index in the score matrix.
            - train_df (pandas.DataFrame): Training positive edges used to generate diffusion scores.
    """
    print(f"\n--- Starting Network Diffusion (Test Ratio={test_ratio}) ---")
    
    # Split Data using existing module
    # This returns train_df (positives) and test_df (positives + negatives with Labels)
    train_df, test_df = create_validation_split(drug_disease_df, test_fraction=test_ratio)
    
    print(f"Train edges: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Prepare Lists of All Nodes
    all_drugs = drug_df['ChemicalName'].unique()
    all_diseases = disease_df['DiseaseName'].unique()
    
    # Run the Math
    Final_Scores, drug_to_idx, disease_to_idx = run_diffusion_math(train_df, all_drugs, all_diseases)
    
    return Final_Scores, test_df, drug_to_idx, disease_to_idx, train_df

def run_diffusion_math(train_df, all_drugs, all_diseases):
    """
    Computes diffusion-based drug–disease scores from training edges using sparse matrix algebra.

    This function implements the core scoring procedure. Given a training set of known
    drug–disease associations, it:
    1. Creates consistent index mappings for drugs and diseases to define matrix coordinates.
    2. Builds a binary sparse adjacency matrix A_train (Disease × Drug) from `train_df`.
       Rows/columns are filtered to include only entities present in `all_diseases` and `all_drugs`.
    3. Computes a Disease–Disease similarity matrix using co-occurrence:
       W_DD = A_train · A_trainᵀ (shared drugs between diseases).
    4. Removes self-loops by clearing the diagonal and eliminating explicit zeros.
    5. Applies L1 row-normalization to W_DD to obtain W_norm (row-stochastic transition-like matrix).
    6. Propagates association evidence via one-step diffusion:
       Final_Scores = W_norm · A_train.

    Args:
        train_df (pandas.DataFrame): Training edge list containing observed links.
            Must include columns 'DiseaseName' and 'ChemicalName'.
        all_drugs (array-like): Complete list/array of drug identifiers (e.g., names).
        all_diseases (array-like): Complete list/array of disease identifiers (e.g., names).

    Returns:
        tuple:
            - Final_Scores: Sparse score matrix (Disease × Drug) containing diffusion scores.
              Higher scores indicate stronger predicted association.
            - drug_to_idx (dict): Mapping from drug identifier -> column index in matrices.
            - disease_to_idx (dict): Mapping from disease identifier -> row index in matrices.
    """
    
    print("Building sparse matrices and running diffusion math...")
    
    # Create consistent mapping
    drug_to_idx = {drug: i for i, drug in enumerate(all_drugs)} 
    disease_to_idx = {disease: i for i, disease in enumerate(all_diseases)}
    
    num_drugs = len(all_drugs)
    num_diseases = len(all_diseases)
    
    # Build A_train (Diseases x Drugs)
    # Filter rows to ensure only use drugs/diseases that exist in valid list
    valid_train = train_df[
        train_df['DiseaseName'].isin(disease_to_idx) & 
        train_df['ChemicalName'].isin(drug_to_idx)
    ]
    
    rows = [disease_to_idx[dis] for dis in valid_train['DiseaseName']]
    cols = [drug_to_idx[drg] for drg in valid_train['ChemicalName']]
    data = [1] * len(rows)
    
    A_train = csr_matrix((data, (rows, cols)), shape=(num_diseases, num_drugs))

    # W_DD = A * A.T (Disease-Disease Similarity)
    # Use float32 to save memory
    W_DD = A_train.astype(np.float32) @ A_train.T.astype(np.float32)
    
    # Remove self-loops
    W_DD.setdiag(0)
    W_DD.eliminate_zeros()

    # Normalize (L1 norm makes it a transition probability matrix)
    W_norm = normalize(W_DD, norm='l1', axis=1)
    
    # Diffusion: Scores = W * A
    Final_Scores = W_norm @ A_train
    
    print("Diffusion math complete.")
    return Final_Scores, drug_to_idx, disease_to_idx