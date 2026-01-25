import networkx as nx
import community as community_louvain # The python-louvain library
import pickle
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd

# Define the absolute path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent

def analyze_disease_network(network_file_path, resolution=1.0):
    """
    Detects disease communities with Louvain and exports community structure artifacts.

    This function runs an end-to-end community analysis workflow by:
    1. Loading a pickled Disease–Disease NetworkX graph from disk.
    2. Running Louvain community detection (weighted) with the given resolution.
       - If the installed Louvain implementation does not support the resolution argument,
         it falls back to the default behavior and resets resolution to 1.0.
    3. Computing modularity and total number of detected communities.
    4. Ranking nodes within each community using Eigenvector Centrality (weighted),
       falling back to weighted degree if eigenvector does not converge.
    5. Building a per-community statistics table including:
       resolution, community ID, size, top hubs, modularity, and total communities.
    6. Saving the full node-to-community partition mapping as a .pkl file for later
       visualization and downstream reporting.
    7. Plotting and saving a log-scale histogram of community sizes.

    Args:
        network_file_path (str or Path): Path to the pickled NetworkX disease graph.
            The graph is expected to be weighted via the 'weight' edge attribute.
        resolution (float): Louvain resolution parameter. Higher values generally
            produce more (smaller) communities. Default is 1.0.

    Returns:
        pandas.DataFrame: A table of community statistics with columns:
            ['Resolution', 'CommunityID', 'Size', 'Top_Hubs', 'Modularity', 'NumCommunities'].
            Returns None if the input graph file does not exist.
    """
    print(f"\n--- Starting Phase 2: Disease Network Analysis ---")
    print(f"Loading disease network from '{network_file_path}'...")
    
    # Load the saved network using pickle
    if not network_file_path.exists():
        print(f"Error: File not found at {network_file_path}")
        return
    
    with open(network_file_path, 'rb') as f:
        disease_net = pickle.load(f)
        
    print(f"Disease network loaded: {disease_net.number_of_nodes()} nodes, {disease_net.number_of_edges()} edges.")
    
    # --- Community Detection using Louvain ---
    print("\nRunning Louvain community detection...")
    start_time = time.time()

    try:
        partition = community_louvain.best_partition(disease_net, weight='weight', resolution=resolution)
    except TypeError:
        print("Warning: 'resolution' not supported by installed library. Using default.")
        partition = community_louvain.best_partition(disease_net, weight='weight')
        resolution = 1.0 # Reset to default if param failed
    
    end_time = time.time()
    modularity = community_louvain.modularity(partition, disease_net, weight='weight')
    num_communities = len(set(partition.values()))
    
    print(f"Louvain algorithm complete! (Took {end_time - start_time:.2f} seconds)")
    print(f"  - Found {num_communities} communities.")
    print(f"  - Modularity score: {modularity:.4f}") 
    
    # --- Analyze Community Sizes ---
    print("Calculating Eigenvector Centrality for node ranking...")

    try:
        # Increase max_iter to ensure convergence on large graphs
        centrality = nx.eigenvector_centrality(disease_net, max_iter=1000, tol=1e-04, weight='weight')
    except:
        print("Eigenvector did not converge, falling back to Degree Centrality.")
        centrality = dict(disease_net.degree(weight='weight'))

    # Group nodes by community and sort by centrality
    community_stats = {}
    for node, comm_id in partition.items():
        if comm_id not in community_stats:
            community_stats[comm_id] = {'Size': 0, 'Nodes': []}
        community_stats[comm_id]['Size'] += 1
        community_stats[comm_id]['Nodes'].append((node, centrality.get(node, 0)))
        
    #prepare data for CSV output
    stats_list = []
    # Sort by size
    sorted_comms = sorted(community_stats.items(), key=lambda x: x[1]['Size'], reverse=True)
    for comm_id, stats in sorted_comms:
        # Get top 5 hubs
        top_nodes = sorted(stats['Nodes'], key=lambda x: x[1], reverse=True)[:5]
        top_names = [n[0] for n in top_nodes]
        
        stats_list.append({
            "Resolution": resolution,
            "CommunityID": comm_id,
            "Size": stats['Size'],
            "Top_Hubs": ", ".join(top_names),
            'Modularity': modularity,        # Adds the modularity score to every row
            'NumCommunities': num_communities # Adds the total count to every row
        })

    # Convert to a DataFrame for easier analysis and sorting
    final_df = pd.DataFrame(stats_list)
    
    print("\nTop 10 largest communities found:")
    # Print just the columns we want to see in the console
    print(final_df[['CommunityID', 'Size', 'Top_Hubs']].head(10))
    
    # Save Partition for Later Visualization
    filename_suffix = f"_res{str(resolution).replace('.', 'p')}"
    results_dir = BASE_DIR / 'results' / 'networks'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f'disease_communities{filename_suffix}.pkl', 'wb') as f:
        pickle.dump(partition, f)
    print(f"\nCommunity assignments saved to '{results_dir / f'disease_communities{filename_suffix}.pkl'}'")

    # Plot Community Size Distribution 
    sizes = [s['Size'] for s in community_stats.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=30, log=True, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title(f'Community Sizes (Resolution={resolution})')
    plt.xlabel('Size')
    plt.ylabel('Frequency (Log Scale)')
    plot_dir = BASE_DIR / 'results' / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'community_dist{filename_suffix}.png', dpi=150)
    plt.close()
    print(f"Community size distribution plot saved to '{plot_dir / f'community_dist{filename_suffix}.png'}'")

    # Return the stats as a DataFrame
    return final_df