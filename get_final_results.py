import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import pickle
import matplotlib.cm as cm
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.data_loader import load_and_clean_data

# Define the absolute path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'data' / 'CTD_chemicals_diseases.csv'
NET_DIR = BASE_DIR / 'results' / 'networks'
TABLES_DIR = BASE_DIR / 'results' / 'tables'
PLOT_DIR = BASE_DIR / 'results' / 'plots'
CENTRALITY_FILE = BASE_DIR / 'results' / 'drug_centrality_scores.csv'
METRICS_FILE = NET_DIR / 'diffusion_metrics.txt'
BEST_RES_FILE = NET_DIR / 'disease_communities_res1p5.pkl'

# Create tables directory if it doesn't exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
NET_DIR.mkdir(parents=True, exist_ok=True)

def process_network_stats():
    """
    Calculates and aggregates topological statistics for all generated networks.

    This function acts as a pipeline health-check and reporting utility by:
    1. Loading the Bipartite (Drug–Disease) graph from disk if available.
    2. Rebuilding and saving the Bipartite graph from the raw dataset if missing.
    3. Loading the projected Drug–Drug and Disease–Disease networks (if present).
    4. Computing core network metrics (Nodes, Edges, Average Degree, Density).
    5. Exporting a consolidated CSV summary table for the final report.

    Notes:
        - The bipartite graph is stored at: NET_DIR / "bipartite_graph.pkl"
        - Projected graphs are expected at:
            * NET_DIR / "drug_network.pkl"
            * NET_DIR / "disease_network.pkl"
        - Average degree is computed as:
            * Bipartite: 2E / N  (global mean degree in an undirected graph)
            * Projected: mean over the node degree list

    Returns:
        None: Saves the results to TABLES_DIR / "network_stats.csv".
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

    This function generates a compact “leaderboard” table for reporting by:
    1. Loading the drug centrality results table from CENTRALITY_FILE.
    2. Sorting drugs in descending order of Betweenness Centrality (bridge importance).
    3. Selecting the key columns needed for the report (DrugName, Betweenness, Eigenvector).
    4. Adding a 1–10 rank column for presentation clarity.
    5. Saving the final table as a CSV in the results tables directory.

    Notes:
        - Expects CENTRALITY_FILE to contain at least the columns:
          'DrugName', 'Betweenness', and 'Eigenvector'.
        - If CENTRALITY_FILE is missing, the function will not crash; it prints a message
          indicating that the Phase 2 centrality computation must be run first.

    Returns:
        None: Saves the results to TABLES_DIR / "top_10_drugs.csv".
    """

    print("\n--- Processing Top 10 Drugs ---")
    if CENTRALITY_FILE.exists():
        # Load Data
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
    Generates and saves the top 10 novel drug–disease link predictions using diffusion scores.

    This function produces a ranked list of high-confidence *unobserved* associations by:
    1. Loading and cleaning the raw Drug–Disease interaction dataset from DATA_FILE.
    2. Building the bipartite adjacency matrix A (Disease × Drug) from known links.
    3. Computing a Disease–Disease similarity matrix W = A·Aᵀ (co-occurrence / shared drugs),
       removing self-similarity (diagonal) and applying L1 row-normalization.
    4. Running a one-step diffusion scoring scheme: Scores = W_norm · A.
    5. Masking existing observed links (setting their scores to zero) so only novel candidates remain.
    6. Extracting the top 10 highest-scoring remaining pairs and exporting them as a CSV table.

    Notes:
        - The prediction pipeline assumes the dataset provides two columns:
          'ChemicalName' (drugs) and 'DiseaseName' (diseases).
        - The adjacency matrix A is built as a binary matrix (1 for observed link).
        - Results are derived from the densified score matrix (Scores.toarray()),
          which may be memory-heavy for very large networks.

    Returns:
        None: Saves the results to TABLES_DIR / "top_10_predictions.csv".
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


def generate_community_report(resolution=1.5):
    """
    Generates a clean, formatted community statistics table for the final report.

    This function prepares a presentation-ready CSV by:
    1. Loading the full community statistics output (disease_community_stats.csv).
    2. Filtering rows to keep only the results for the specified Louvain resolution.
    3. Sorting communities by size (largest communities first).
    4. Selecting the key columns needed for reporting, and renaming them to
       human-friendly headers.
    5. Saving the final report table as final_report_table.csv and printing a quick
       console preview (top 10 communities).

    Args:
        resolution (float): Louvain resolution value to extract (e.g., 1.0, 1.2, 1.5).

    Notes:
        - Expects the input CSV to contain at least these columns:
          'Resolution', 'CommunityID', 'Size', 'Modularity', 'NumCommunities', 'Top_Hubs'.
        - Filtering is done using direct float equality (df['Resolution'] == resolution).
          If the stored values have float rounding issues, consider using a tolerance-based
          comparison (e.g., np.isclose).

    Returns:
        None: Saves the results to BASE_DIR / "results/tables/final_report_table.csv".
    """

    input_path = BASE_DIR / 'results' / 'tables' / 'disease_community_stats.csv'
    output_path = BASE_DIR / 'results' / 'tables' / 'final_report_table.csv'
    
    print(f"\n--- Generating Report Table (Resolution={resolution}) ---")
    
    if not input_path.exists():
        print(f"Error: Stats file not found at {input_path}")
        return

    # Load Data
    df = pd.read_csv(input_path)
    
    # Filter for the Target Resolution
    # Using a small epsilon for float comparison to be safe, or direct equality
    subset = df[df['Resolution'] == resolution].copy()
    
    if subset.empty:
        print(f"Warning: No data found for resolution {resolution}.")
        print(f"Available resolutions: {df['Resolution'].unique()}")
        return

    # Sort by Size (Largest Communities First)
    subset = subset.sort_values(by='Size', ascending=False)
    
    # Select and Rename Columns for the Report
    final_table = subset[['CommunityID', 'Size', 'Modularity', 'NumCommunities','Top_Hubs']]
    final_table.columns = ['Community ID', 'Num. Diseases', 'Modularity', 'Total Communities', 'Top Central Hubs (Eigenvector)']

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_table.to_csv(output_path, index=False)
    
    # Print Top 10 to Console (for quick check)
    print(f"Success! Table saved to: {output_path}")
    print("\n--- PREVIEW: Top 10 Communities ---")
    print(final_table.head(10).to_string(index=False))
    print("-" * 60)


def generate_zipf_comparison():
    """
    Generates a comparative Rank–Size (Zipf) plot of community structures across all resolutions.

    This function visualizes how the Louvain resolution parameter affects the distribution of
    community sizes by:
    1. Discovering all saved disease community partition files matching:
       NET_DIR / "disease_communities_res*.pkl".
    2. Loading each partition (node -> community ID mapping) from disk.
    3. Computing community sizes via frequency counts of community IDs.
    4. Sorting community sizes in descending order and assigning ranks (1..k).
    5. Plotting all resolutions on the same log–log Rank vs Size chart for comparison.
    6. Saving the resulting figure to the plots directory.

    Notes:
        - Each PKL file is expected to store a partition dictionary: {node: community_id}.
        - The plot uses log–log scaling (plt.loglog) to highlight heavy-tailed behavior.
        - Styling (linestyle/marker) cycles to keep multiple resolutions visually distinct.

    Returns:
        None: Saves the figure to PLOT_DIR / "community_zipf_comparison.png".
    """

    print(f"\n--- [3/3] Generating Comparative Zipf Plot ---")
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all community files
    comm_files = sorted(list(NET_DIR.glob("disease_communities_res*.pkl")))
    
    if not comm_files:
        print("Error: No community files found in results/networks.")
        return

    plt.figure(figsize=(10, 8))
    
    # Define styles for distinction
    styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    # Loop through files and plot
    for i, comm_file in enumerate(comm_files):
        res_name = comm_file.stem.split('_')[-1].replace('p', '.') 
        
        with open(comm_file, 'rb') as f:
            partition = pickle.load(f)
            
        # Get sizes and sort them
        counts = pd.Series(list(partition.values())).value_counts()
        sorted_sizes = counts.sort_values(ascending=False).values
        ranks = np.arange(1, len(sorted_sizes) + 1)
        
        # Plot
        plt.loglog(ranks, sorted_sizes, 
                   linestyle=styles[i % 3], marker=markers[i % 3], 
                   linewidth=2, markersize=6, alpha=0.7,
                   label=f"Resolution {res_name} (n={len(sorted_sizes)})")

    # Formatting
    plt.title("Impact of Resolution on Community Structure (Zipf Plot)", fontsize=16)
    plt.xlabel("Rank (Log Scale)", fontsize=14)
    plt.ylabel("Community Size (Log Scale)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    # Save
    output_path = PLOT_DIR / "community_zipf_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success! Comparative Zipf plot saved to {output_path}")


def generate_comparison_plots():
    """
    Generates comparative Spring Layout and UMAP plots for all available Louvain resolutions.

    This function produces a consistent, high-contrast visualization set to compare how the
    disease-community structure changes with the resolution parameter by:
    1. Loading a pruned Disease–Disease network (for readable plotting) and precomputed
       node embeddings (for UMAP projection).
    2. Computing and fixing the geometric layouts once:
       - A force-directed Spring layout (NetworkX) for graph geometry.
       - A 2D UMAP projection from the embedding vectors for feature-space geometry.
       These fixed layouts ensure that only community coloring changes between resolutions.
    3. Discovering all saved community partitions matching:
       NET_DIR / "disease_communities_res*.pkl".
    4. For each resolution:
       - Creating a Spring Layout plot where node colors are assigned deterministically
         via a high-contrast palette (tab10) using (community_id % 10).
       - Building a legend for the Top 10 largest communities (by size) using the same
         color mapping to ensure exact consistency.
       - Creating a UMAP scatter plot restricted to the Top 10 communities, using the
         exact same palette mapping as the Spring plot.

    Notes:
        - Expects the following input files:
            * NET_DIR / "disease_network_pruned_p90.pkl"  (Graph for visualization)
            * NET_DIR / "disease_embeddings_fast.pkl"     (node -> embedding vector)
            * NET_DIR / "disease_communities_res*.pkl"    (partition dicts)
        - Community partitions are expected as dictionaries: {node: community_id}.
        - Colors are intentionally deterministic and resolution-independent:
          color = tab10[community_id % 10]. If communities exceed 10, colors will cycle.
        - The UMAP plot is filtered to Top 10 communities to keep the plot readable.

    Returns:
        None: Saves figures to PLOT_DIR as:
            - "compare_spring_<res_name>.png"
            - "compare_umap_<res_name>.png"
    """

    print(f"\n--- [2/2] Generating Comparative Plots ---")
    
    pruned_net_path = NET_DIR / 'disease_network_pruned_p90.pkl'
    embeddings_path = NET_DIR / 'disease_embeddings_fast.pkl'
    
    # Check Files
    if not pruned_net_path.exists() or not embeddings_path.exists():
        print("Error: Pruned network or embeddings missing. Run Phase 3 first.")
        return

    # Load Geometry
    print("Loading Graph and Embeddings...")
    with open(pruned_net_path, 'rb') as f:
        G = pickle.load(f)
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    # Calculate Layouts (Shape)
    print("Calculating Layouts (Shape)...")
    print("   > Spring Layout (Force-Directed)...")
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    print("   > UMAP 2D Projection...")
    embedded_nodes = list(embeddings.keys())
    embedding_vectors = np.array([embeddings[node] for node in embedded_nodes])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embedding_vectors)
    
    umap_map = {node: (embedding_2d[i, 0], embedding_2d[i, 1]) for i, node in enumerate(embedded_nodes)}

    # Loop Through Resolutions
    comm_files = sorted(list(NET_DIR.glob("disease_communities_res*.pkl")))
    if not comm_files:
        print("Error: No community files found.")
        return

    print(f"Generating plots for {len(comm_files)} resolutions found...")

    # --- COLOR SELECTION ---
    # Used 'Set1' (9 colors) or 'tab10' (10 colors) for maximum contrast.
    # Convert the colormap to a fixed list of RGB tuples.
    palette_colors = cm.get_cmap('tab10').colors 

    for comm_file in comm_files:
        res_name = comm_file.stem.split('_')[-1] 
        print(f"   > Processing {res_name}...")
        
        with open(comm_file, 'rb') as f:
            partition = pickle.load(f)
        
        # Spring Layout 
        plt.figure(figsize=(14, 12))
        
        # Assign distinct color to every node based on ID % 10
        node_rgb_colors = [palette_colors[partition.get(n, 0) % 10] for n in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=20, 
                               node_color=node_rgb_colors, 
                               alpha=0.9) # Increased alpha for better visibility
        nx.draw_networkx_edges(G, pos, alpha=0.03, edge_color='gray')
        
        # Manual Legend
        import matplotlib.lines as mlines
        
        comm_counts = pd.Series(list(partition.values())).value_counts()
        top_10_ids = comm_counts.head(10).index.tolist()
        
        legend_handles = []
        for comm_id in top_10_ids:
            # EXACT same color logic: ID % 10
            color = palette_colors[comm_id % 10]
            
            handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                  markersize=12, label=f'ID {comm_id} (n={comm_counts[comm_id]})')
            legend_handles.append(handle)
            
        plt.legend(handles=legend_handles, title="Top 10 Communities", 
                   bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
        
        plt.title(f"Spring Layout ({res_name.replace('p', '.')})", fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"compare_spring_{res_name}.png", dpi=300) 
        plt.close()

        # UMAP Plot
        plot_data = []
        for node, (x, y) in umap_map.items():
            plot_data.append({'x': x, 'y': y, 'community': partition.get(node, -1)})
        df = pd.DataFrame(plot_data)
        
        # Filter for top 10
        df_filtered = df[df['community'].isin(top_10_ids)].copy()
        
        plt.figure(figsize=(12, 10))
        
        # Create explicit palette dictionary for Seaborn
        custom_palette = {cid: palette_colors[cid % 10] for cid in top_10_ids}
        
        sns.scatterplot(data=df_filtered, x='x', y='y', hue='community', 
                        palette=custom_palette, 
                        s=50, alpha=0.8, 
                        legend='full') 
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Community ID", fontsize=12)
        plt.title(f"UMAP Projection ({res_name.replace('p', '.')}) - Top 10 Communities", fontsize=16)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"compare_umap_{res_name}.png", dpi=300)
        plt.close()

    print(f"Done! High-contrast plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    # comment and uncomment functions as needed
    process_network_stats()
    save_top_drugs()
    save_predictions()
    generate_community_report(resolution=1.5)
    generate_zipf_comparison()
    generate_comparison_plots()