# src/network_builder.py
import networkx as nx
from networkx.algorithms import bipartite
import time
import pickle

def build_and_project_networks(edge_list_df, output_dir):
    """
    Builds and projects the bipartite graph, saving the results to the specified directory.
    
    Args:
        edge_list_df (pandas.DataFrame): DataFrame of drug-disease edges.
        output_dir (pathlib.Path): The directory to save the network files in.
        
    Returns:
        tuple: A tuple containing the disease_network and drug_network.
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
    
    # Use write_gpickle for faster binary saving
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