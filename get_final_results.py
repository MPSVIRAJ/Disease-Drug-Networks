import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.data_loader import load_and_clean_data

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'data' / 'CTD_chemicals_diseases.csv'
NET_DIR = BASE_DIR / 'results' / 'networks'
TABLES_DIR = BASE_DIR / 'results' / 'tables'
CENTRALITY_FILE = BASE_DIR / 'results' / 'drug_centrality_scores.csv'
METRICS_FILE = NET_DIR / 'diffusion_metrics.txt'

# Create tables directory if it doesn't exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
NET_DIR.mkdir(parents=True, exist_ok=True)

def process_network_stats():
    """
    Calculates and aggregates topological statistics for all generated networks.

    This function performs a health check on the network analysis pipeline by:
    1. Reconstructing the Bipartite (Drug-Disease) graph if missing.
    2. Loading the Projected Drug-Drug and Disease-Disease networks.
    3. Computing key metrics (Node count, Edge count, Average Degree, Density).
    4. Compiling the results into a summary CSV for reporting.

    Returns:
        None: Saves the results to 'results/tables/network_stats.csv'.
    """
    print("\n--- Processing Network Statistics ---")
    stats_data = []

    # Handle Bipartite Graph
    bipartite_path = NET_DIR / "bipartite_graph.pkl"
    
    if bipartite_path.exists():
        print(f"Loading existing bipartite graph from {bipartite_path}...")
        with open(bipartite_path, 'rb') as f:
            G_bi = pickle.load(f)
    else:
        print("Bipartite graph not found. Rebuilding and saving...")
        if DATA_FILE.exists():
            df = load_and_clean_data(DATA_FILE)
            G_bi = nx.Graph()
            drugs = df['ChemicalName'].unique()
            diseases = df['DiseaseName'].unique()
            G_bi.add_nodes_from(drugs, bipartite=0)
            G_bi.add_nodes_from(diseases, bipartite=1)
            G_bi.add_edges_from([tuple(x) for x in df.values])
            
            # Save it
            with open(bipartite_path, 'wb') as f:
                pickle.dump(G_bi, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved bipartite graph to {bipartite_path}")
        else:
            print("Error: Data file missing, cannot rebuild bipartite graph.")
            return

    # Calculate Bipartite Stats
    stats_data.append({
        "Network Type": "Bipartite (Drug-Disease)",
        "Nodes": G_bi.number_of_nodes(),
        "Edges": G_bi.number_of_edges(),
        "Avg Degree": 2 * G_bi.number_of_edges() / G_bi.number_of_nodes() if G_bi.number_of_nodes() > 0 else 0,
        "Density": nx.density(G_bi)
    })

    # Process Projected Networks
    files = {
        "Projected (Drug-Drug)": "drug_network.pkl",
        "Projected (Disease-Disease)": "disease_network.pkl"
    }
    
    for name, fname in files.items():
        fpath = NET_DIR / fname
        if fpath.exists():
            with open(fpath, 'rb') as f:
                G = pickle.load(f)
            
            # Avg Degree calculation for unipartite
            degrees = [d for n, d in G.degree()]
            avg_deg = sum(degrees) / len(degrees) if degrees else 0
            
            stats_data.append({
                "Network Type": name,
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Avg Degree": avg_deg,
                "Density": nx.density(G)
            })
        else:
            print(f"Warning: {fname} not found.")

    # Save Stats Table
    df_stats = pd.DataFrame(stats_data)
    output_path = TABLES_DIR / "network_stats.csv"
    df_stats.to_csv(output_path, index=False)
    print(f"Saved network statistics to {output_path}")
    print(df_stats)

def save_top_drugs():
    """
    Extracts and exports the top 10 most influential drugs based on Betweenness Centrality.

    This function reads the comprehensive centrality scores generated in the analysis
    phase, sorts the drugs by their Betweenness score (indicating their role as
    bridges in the network), and saves a simplified leaderboard.

    Returns:
        None: Saves the results to 'results/tables/top_10_drugs.csv'.
    """

    print("\n--- Processing Top 10 Drugs ---")
    if CENTRALITY_FILE.exists():
        df = pd.read_csv(CENTRALITY_FILE)
        # Sort by Betweenness and take top 10
        top_10 = df.sort_values(by='Betweenness', ascending=False).head(10)
        
        # Select and Rename columns for clean output
        output_df = top_10[['DrugName', 'Betweenness', 'Eigenvector']]
        output_df.insert(0, 'Rank', range(1, 11))
        
        output_path = TABLES_DIR / "top_10_drugs.csv"
        output_df.to_csv(output_path, index=False)
        print(f"Saved top 10 drugs to {output_path}")
        print(output_df)
    else:
        print("Centrality file not found. Run Phase 2 first.")

def save_predictions():
    """
    Generates and saves the top 10 novel drug-disease link predictions.

    This function performs a lightweight re-run of the diffusion process to:
    1. Reconstruct the similarity and adjacency matrices.
    2. Compute diffusion scores.
    3. Mask (zero out) edges that already exist in the training data.
    4. Identify the highest-scoring unobserved associations.

    Returns:
        None: Saves the results to 'results/tables/top_10_predictions.csv'.
    """
    print("\n--- Processing Top 10 Predictions ---")
    if not DATA_FILE.exists():
        print("Data file missing, cannot run predictions.")
        return

    # Load Data
    df = load_and_clean_data(DATA_FILE)
    all_drugs = sorted(df['ChemicalName'].unique())
    all_diseases = sorted(df['DiseaseName'].unique())
    
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    dis_to_idx = {d: i for i, d in enumerate(all_diseases)}
    
    # Build Matrices
    rows = [dis_to_idx[d] for d in df['DiseaseName']]
    cols = [drug_to_idx[d] for d in df['ChemicalName']]
    data = [1] * len(rows)
    
    # Adjacency (Disease x Drug)
    A = csr_matrix((data, (rows, cols)), shape=(len(all_diseases), len(all_drugs)))
    
    # Similarity (Disease x Disease)
    W = A.astype(np.float32) @ A.T.astype(np.float32)
    W.setdiag(0)
    W_norm = normalize(W, norm='l1', axis=1)
    
    # Predict Scores
    print("Calculating scores (this may take a moment)...")
    Scores = W_norm @ A
    Scores_dense = Scores.toarray()
    
    # Filter: Zero out existing links
    existing_indices = A.nonzero()
    Scores_dense[existing_indices] = 0
    
    # Find Top 10
    flat_indices = np.argsort(Scores_dense.ravel())[-10:][::-1]
    
    results = []
    for rank, idx in enumerate(flat_indices, 1):
        dis_idx, drg_idx = np.unravel_index(idx, Scores_dense.shape)
        results.append({
            "Rank": rank,
            "Drug": all_drugs[drg_idx],
            "Predicted Disease": all_diseases[dis_idx],
            "Score": round(Scores_dense[dis_idx, drg_idx], 4)
        })
        
    df_preds = pd.DataFrame(results)
    output_path = TABLES_DIR / "top_10_predictions.csv"
    df_preds.to_csv(output_path, index=False)
    print(f"Saved top 10 predictions to {output_path}")
    print(df_preds)

if __name__ == "__main__":
    process_network_stats()
    save_top_drugs()
    save_predictions()