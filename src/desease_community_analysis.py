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
    """Loads the disease network and performs community detection."""
    print(f"\n--- Starting Phase 2: Disease Network Analysis ---")
    print(f"Loading disease network from '{network_file_path}'...")
    
    # Load the saved network using pickle
    with open(network_file_path, 'rb') as f:
        disease_net = pickle.load(f)
        
    print(f"Disease network loaded: {disease_net.number_of_nodes()} nodes, {disease_net.number_of_edges()} edges.")
    
    # --- Community Detection using Louvain ---
    print("\nRunning Louvain community detection...")
    start_time = time.time()
    
    # The Louvain algorithm works best on unweighted graphs for partitioning,
    # but can handle weights to measure modularity. We'll use weights here.
    # It returns a dictionary: {node: community_id}
    partition = community_louvain.best_partition(disease_net, weight='weight')
    
    end_time = time.time()
    modularity = community_louvain.modularity(partition, disease_net, weight='weight')
    num_communities = len(set(partition.values()))
    
    print(f"Louvain algorithm complete! (Took {end_time - start_time:.2f} seconds)")
    print(f"  - Found {num_communities} communities.")
    print(f"  - Modularity score: {modularity:.4f}") # Higher is better structured
    
    # --- Analyze Community Sizes ---
    community_sizes = {}
    for node, comm_id in partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
    # Convert to a DataFrame for easier analysis and sorting
    size_df = pd.DataFrame(community_sizes.items(), columns=['CommunityID', 'Size'])
    size_df = size_df.sort_values(by='Size', ascending=False).reset_index(drop=True)
    
    print("\nTop 10 largest communities found:")
    print(size_df.head(10))
    
    # --- Optional: Save Partition for Later Visualization ---
    partition_file = BASE_DIR / 'results' / 'networks' / 'disease_communities.pkl'
    with open(partition_file, 'wb') as f:
        pickle.dump(partition, f)
    print(f"\nCommunity assignments saved to '{partition_file}'")

    # --- Optional: Plot Community Size Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(size_df['Size'], bins=max(50, num_communities // 10), log=True)
    plt.title('Distribution of Community Sizes (Log Scale)')
    plt.xlabel('Community Size')
    plt.ylabel('Frequency (Log Scale)')
    plot_path = BASE_DIR / 'results' / 'plots' / 'community_size_distribution.png'
    plt.savefig(plot_path)
    print(f"Community size distribution plot saved to '{plot_path}'")
    # plt.show() # Uncomment if you want to see the plot immediately