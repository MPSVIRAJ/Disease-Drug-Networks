# src/network_builder.py
import networkx as nx
from networkx.algorithms import bipartite
import time

def build_and_project_networks(edge_list_df):
    """
    Builds a bipartite graph and projects it into disease-disease and drug-drug networks.
    
    Args:
        edge_list_df (pandas.DataFrame): DataFrame with 'ChemicalName' and 'DiseaseName' columns.
        
    Returns:
        tuple: A tuple containing the disease_network and drug_network (networkx.Graph objects).
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
    
    # Save the networks for later use
    nx.write_gexf(disease_network, "results/networks/disease_network.gexf")
    nx.write_gexf(drug_network, "results/networks/drug_network.gexf")
    
    print("\n--- Projected Networks Saved ---")
    print(f"Disease-Disease network saved to 'results/networks/disease_network.gexf'")
    print(f"Drug-Drug network saved to 'results/networks/drug_network.gexf'")
    
    return disease_network, drug_network