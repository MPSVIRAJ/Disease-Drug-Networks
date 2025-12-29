import networkx as nx
import pickle
from pathlib import Path
import time
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

def analyze_drug_network(network_file_path):
    """
    Calculates centrality measures for a drug-drug interaction network.

    This function loads a pre-built network graph and computes three key centrality 
    metrics to identify influential drugs:
    1. Weighted Degree (Strength): Measures the total weight of connections.
    2. Betweenness Centrality: Approximated using k=1000 samples to ensure computational 
    efficiency on large graphs.
    3. Eigenvector Centrality: Computed on the unweighted graph to measure node influence.

    The function aggregates these scores into a DataFrame, prints the top 10 drugs for 
    each metric, and exports the full results to a CSV file.

    Args:
        network_file_path (str or Path): The file path to the pickled NetworkX 
            graph object.

    Returns:
        None: Results are saved to 'results/drug_centrality_scores.csv'.
    """
    print(f"\n--- Starting Drug Network Analysis ---")
    print(f"Loading drug network from '{network_file_path}'...")
    
    # Load the saved network
    with open(network_file_path, 'rb') as f:
        drug_net = pickle.load(f)
        
    print(f"Drug network loaded: {drug_net.number_of_nodes()} nodes, {drug_net.number_of_edges()} edges.")
    
    # --- Centrality Measures Calculation ---
    # Weighted Degree (Strength)
    print("\nCalculating Weighted Degree (Strength)...")
    start_time = time.time()
    strength = dict(drug_net.degree(weight='weight'))
    end_time = time.time()
    print(f"Strength calculation complete. (Took {end_time - start_time:.2f} seconds)")

    # Betweenness Centrality
    # NOTE: Calculating weighted betweenness centrality on a graph this large can be
    # extremely time-consuming (hours or days). We'll calculate the unweighted version
    # for speed as a good approximation for this project scope. Add k= parameter to speed up.
    # Set k to sample ~10% of nodes for approximation if needed: k = drug_net.number_of_nodes() // 10
    
    print("\nCalculating Betweenness Centrality (Unweighted Approximation)...")
    start_time = time.time()
    
    betweenness = nx.betweenness_centrality(drug_net, k=1000, normalized=True, weight=None)
    end_time = time.time()
    print(f"Betweenness calculation complete. (Took {end_time - start_time:.2f} seconds)")

    # Eigenvector Centrality
    # Here use the unweighted version for robustness and speed up the computation.
    print("\nCalculating Eigenvector Centrality (Unweighted)...")
    start_time = time.time()
    
    try:
        eigenvector = nx.eigenvector_centrality(drug_net, max_iter=1000, tol=1e-04)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge. Try increasing max_iter or using alternative methods.")
        eigenvector = {node: 0.0 for node in drug_net.nodes()} # Assign default value if fails
    end_time = time.time()
    print(f"Eigenvector calculation complete. (Took {end_time - start_time:.2f} seconds)")

    # Combine Results into a DataFrame 
    centrality_df = pd.DataFrame({
        'DrugName': list(drug_net.nodes()),
        'Strength': [strength.get(node, 0) for node in drug_net.nodes()],
        'Betweenness': [betweenness.get(node, 0.0) for node in drug_net.nodes()],
        'Eigenvector': [eigenvector.get(node, 0.0) for node in drug_net.nodes()]
    })

    # Display Top Ranked Drugs 
    print("\n--- Top 10 Drugs by Strength ---")
    print(centrality_df.sort_values(by='Strength', ascending=False).head(10))
    
    print("\n--- Top 10 Drugs by Betweenness Centrality ---")
    print(centrality_df.sort_values(by='Betweenness', ascending=False).head(10))
    
    print("\n--- Top 10 Drugs by Eigenvector Centrality ---")
    print(centrality_df.sort_values(by='Eigenvector', ascending=False).head(10))
    
    # Save Centrality Results 
    results_file = BASE_DIR / 'results' / 'drug_centrality_scores.csv'
    centrality_df.to_csv(results_file, index=False)
    print(f"\nCentrality scores saved to '{results_file}'")