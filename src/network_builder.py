# src/network_builder.py
import networkx as nx
from networkx.algorithms import bipartite
import time
import pickle
from pathlib import Path
import numpy as np

def build_and_project_networks(edge_list_df, output_dir):
    """
    Constructs a bipartite graph from edge lists and projects it into unipartite networks.

    This function creates a bipartite graph containing both Drug and Disease nodes.
    It then performs weighted bipartite projections to create:
    1. A Disease-Disease network (edges weighted by number of shared drugs).
    2. A Drug-Drug network (edges weighted by number of shared diseases).
    The resulting graphs are serialized and saved as pickle files.

    Args:
        edge_list_df (pd.DataFrame): DataFrame containing columns 'ChemicalName' 
            and 'DiseaseName' representing the edges.
        output_dir (Path): The directory path where the projected network pickle 
            files will be saved.

    Returns:
        tuple[nx.Graph, nx.Graph]: A tuple containing (disease_network, drug_network).
    """

    print("\nBuilding the bipartite graph...")
    B = nx.Graph()
    drug_nodes = edge_list_df['ChemicalName'].unique()
    disease_nodes = edge_list_df['DiseaseName'].unique()
    
    B.add_nodes_from(drug_nodes, bipartite=0)
    B.add_nodes_from(disease_nodes, bipartite=1)
    B.add_edges_from([tuple(x) for x in edge_list_df.values])
    
    print("Projecting into Disease-Disease and Drug-Drug networks...")
    start_time = time.time()
    
    disease_network = bipartite.weighted_projected_graph(B, disease_nodes)
    drug_network = bipartite.weighted_projected_graph(B, drug_nodes)
    
    end_time = time.time()
    print(f"Projection complete! (Took {end_time - start_time:.2f} seconds)")
    
    print("\nSaving projected networks using pickle...")
    start_save_time = time.time()
    
    disease_net_path = output_dir / "disease_network.pkl"
    drug_net_path = output_dir / "drug_network.pkl"
    
    with open(disease_net_path, 'wb') as f:
        pickle.dump(disease_network, f, pickle.HIGHEST_PROTOCOL)
        
    with open(drug_net_path, 'wb') as f:
        pickle.dump(drug_network, f, pickle.HIGHEST_PROTOCOL)
        
    end_save_time = time.time()
    print(f"Saving complete! (Took {end_save_time - start_save_time:.2f} seconds)")
    
    print("\n--- Projected Networks Saved ---")
    print(f"Disease-Disease network saved to '{disease_net_path}'")
    print(f"Drug-Drug network saved to '{drug_net_path}'")
    
    return disease_network, drug_network


def prune_network_by_weight(input_network_path, output_network_path, percentile=90):
    """
    Loads a network and removes edges falling below a specified weight percentile.

    This function helps reduce noise and computational complexity by keeping only 
    the strongest associations. It calculates a weight threshold based on the 
    input percentile, removes weaker edges, and subsequently removes any nodes 
    that become isolated.

    Args:
        input_network_path (Path): Path to the source NetworkX graph pickle file.
        output_network_path (Path): Destination path for the pruned graph pickle file.
        percentile (int, optional): The percentile threshold (0-100) for edge weights. 
            Edges with weights below this value are removed. Defaults to 90.

    Returns:
        nx.Graph: The pruned NetworkX graph object. Returns None if the input 
        file is not found.
    """
    print(f"\n--- Pruning Network ---")
    print(f"Loading network from '{input_network_path}'...")
    try:
        with open(input_network_path, 'rb') as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Network file not found at {input_network_path}")
        return None

    if G.number_of_edges() == 0:
        print("Network has no edges to prune.")

        with open(output_network_path, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        print(f"Empty network saved to '{output_network_path}'")
        return G

    print(f"Original network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    weights = [d.get('weight', 0) for u, v, d in G.edges(data=True)] 
    if not weights:
        print("No edge weights found or all weights are zero. Cannot prune by weight.")

        with open(output_network_path, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        print(f"Original network saved (no weights) to '{output_network_path}'")
        return G

    try:
      threshold = np.percentile(weights, percentile)
    except IndexError:

      print("Warning: Could not calculate percentile, possibly too few edges. Using minimum weight as threshold.")
      threshold = min(weights) if weights else 0

    print(f"Pruning edges with weight < {threshold:.4f} (keeping edges >= {percentile}th percentile)")

    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(G.nodes()) 

    edges_to_keep = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('weight', 0) >= threshold]
    G_pruned.add_edges_from(edges_to_keep)

    # Remove isolated nodes after pruning 
    isolated_nodes = list(nx.isolates(G_pruned))
    if isolated_nodes:
        print(f"Removing {len(isolated_nodes)} isolated nodes after pruning.")
        G_pruned.remove_nodes_from(isolated_nodes)

    print(f"Pruned network: {G_pruned.number_of_nodes()} nodes, {G_pruned.number_of_edges()} edges.")

    # Save the pruned network
    with open(output_network_path, 'wb') as f:
        pickle.dump(G_pruned, f, pickle.HIGHEST_PROTOCOL)
    print(f"Pruned network saved to '{output_network_path}'")
    return G_pruned