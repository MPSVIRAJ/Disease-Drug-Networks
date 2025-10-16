import pandas as pd

def load_and_clean_data(file_path='data/CTD_chemicals_diseases.csv'):
    """
    Load and clean the CTD chemicals-diseases dataset.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with relevant columns for network creation.
    """
    print(f"Loading data from '{file_path}'...")
    df = pd.read_csv(file_path, comment='#', header=None)

    column_names = [
        'ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID',
        'DirectEvidence', 'InferenceGeneSymbol', 'InferenceScore', 'OmimIDs', 'PubMedIDs'
    ]
    df.columns = column_names

    print("Data loaded. Preparing data for network creation...")
    edge_list_df = df[['ChemicalName', 'DiseaseName']].dropna().drop_duplicates()

    print("\n--- Data Preparation Complete! ---")
    return edge_list_df
