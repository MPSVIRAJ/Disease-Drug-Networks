import networkx as nx
import community as community_louvain # The python-louvain library
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd

# Define the absolute path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent

def analyze_disease_network(network_file_path):
    """
    Loads a disease network graph and identifies communities using the Louvain algorithm.

    This function performs a complete analysis pipeline:
    1. Loads a NetworkX graph from a specific pickle file.
    2. Applies the Louvain method (`best_partition`) to detect communities based on edge weights.
    3. Calculates the modularity score to quantify the quality of the division.
    4. Aggregates and reports the size of the top 10 communities.
    5. Saves the resulting partition dictionary to disk for later use.
    6. Generates and saves a log-scale histogram of community sizes.

    Args:
        network_file_path (str or Path): The file path to the pickled NetworkX 
            graph object representing the disease network.

    Returns:
        None: This function operates via side effects (printing statistics, saving 
        a .pkl file, and saving a .png plot).
    """
    print(f"\n--- Starting Phase 2: Disease Network Analysis ---")
    print(f"Loading disease network from '{network_file_path}'...")
    
    # Load the saved network using pickle
    with open(network_file_path, 'rb') as f:
        disease_net = pickle.load(f)
        
    print(f"Disease network loaded: {disease_net.number_of_nodes()} nodes, {disease_net.number_of_edges()} edges.")
    
    # --- Community Detection using Louvain ---
    print("\nRunning Louvain community detection...")
    start_time = time.time()

    partition = community_louvain.best_partition(disease_net, weight='weight')
    
    end_time = time.time()
    modularity = community_louvain.modularity(partition, disease_net, weight='weight')
    num_communities = len(set(partition.values()))
    
    print(f"Louvain algorithm complete! (Took {end_time - start_time:.2f} seconds)")
    print(f"  - Found {num_communities} communities.")
    print(f"  - Modularity score: {modularity:.4f}") 
    
    # --- Analyze Community Sizes ---
    community_sizes = {}
    for node, comm_id in partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
    # Convert to a DataFrame for easier analysis and sorting
    size_df = pd.DataFrame(community_sizes.items(), columns=['CommunityID', 'Size'])
    size_df = size_df.sort_values(by='Size', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 largest communities found:")
    print(size_df.head(10))
    
    # Save Partition for Later Visualization
    partition_file = BASE_DIR / 'results' / 'networks' / 'disease_communities.pkl'
    with open(partition_file, 'wb') as f:
        pickle.dump(partition, f)
    print(f"\nCommunity assignments saved to '{partition_file}'")

    # Plot Community Size Distribution 
    plt.figure(figsize=(10, 6))
    plt.hist(size_df['Size'], bins=max(50, num_communities // 10), log=True, color='skyblue', edgecolor='black')
    
    plt.title('Distribution of Community Sizes', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Community Size (Number of Diseases)', fontsize=16)
    plt.ylabel('Frequency (Log Scale)', fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    plt.tight_layout()
    plot_path = BASE_DIR / 'results' / 'plots' / 'community_size_distribution.png'
    plt.savefig(plot_path)
    print(f"Community size distribution plot saved to '{plot_path}'")