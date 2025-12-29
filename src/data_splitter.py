# src/data_splitter.py
import pandas as pd
import random

def create_validation_split(edge_list_df, test_fraction=0.1):
    """
    Splits an edge list into training and validation sets for link prediction.

    This function partitions the known positive associations into a training set and a 
    test set. It also generates a set of negative samples (fabricated non-associations) 
    equal in size to the positive test set. The resulting test DataFrame includes both 
    positive (Label=1) and negative (Label=0) samples to evaluate binary classification 
    performance.

    Args:
        edge_list_df (pd.DataFrame): A DataFrame containing the known positive edges. 
            It must have two columns representing the source and target nodes (e.g., 
            ['ChemicalName', 'DiseaseName']).
        test_fraction (float, optional): The proportion of the dataset to include in 
            the test split. Defaults to 0.1 (10%).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - train_df: Contains the training edges (positive samples only) with 
            columns ['ChemicalName', 'DiseaseName'].
            - test_df: Contains both positive and negative edges for validation, 
            with an additional 'Label' column (1 for positive, 0 for negative).
    """
    print("\n--- Creating Validation Split ---")
    
    # Get the positive links (all known associations)
    positive_edges = [tuple(x) for x in edge_list_df.values]
    random.shuffle(positive_edges)
    
    # Split positive links into train and test
    test_size = int(len(positive_edges) * test_fraction)
    test_positive_edges = positive_edges[:test_size]
    train_edges = positive_edges[test_size:]
    
    print(f"Total associations: {len(positive_edges)}")
    print(f"Training set size: {len(train_edges)} associations")
    print(f"Test set size (positive): {len(test_positive_edges)} associations")
    
    # Create negative samples for the test set
    print("Generating negative samples for the test set...")
    drug_nodes = set(edge_list_df['ChemicalName'])
    disease_nodes = set(edge_list_df['DiseaseName'])
    all_positive_edges_set = set(positive_edges) # For fast lookup
    
    test_negative_edges = []
    while len(test_negative_edges) < test_size:
        # Pick a random drug and a random disease
        rand_drug = random.choice(list(drug_nodes))
        rand_disease = random.choice(list(disease_nodes))
        
        pair = (rand_drug,rand_disease)
        
        # Add it to negative set ONLY if it's not a known positive link
        if pair not in all_positive_edges_set:
            test_negative_edges.append(pair)
            
    print(f"Test set size (negative): {len(test_negative_edges)} associations")

    # Create DataFrames for train/test
    train_df = pd.DataFrame(train_edges, columns=['ChemicalName', 'DiseaseName'])
    
    # Create test DataFrame with labels (1=positive, 0=negative)
    test_pos_df = pd.DataFrame(test_positive_edges, columns=['ChemicalName', 'DiseaseName'])
    test_pos_df['Label'] = 1
    test_neg_df = pd.DataFrame(test_negative_edges, columns=['ChemicalName', 'DiseaseName'])
    test_neg_df['Label'] = 0
    
    test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True)
    
    print("Validation split complete.")
    return train_df, test_df