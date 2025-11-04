import networkx as nx
import pickle
from pathlib import Path
import time
import pandas as pd

# Define BASE_DIR relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

def analyze_drug_network(network_file_path):
    """Loads the drug network and calculates centrality measures."""
    print(f"\n--- Starting Drug Network Analysis ---")
    print(f"Loading drug network from '{network_file_path}'...")
    
    # Load the saved network
    with open(network_file_path, 'rb') as f:
        drug_net = pickle.load(f)
        
    print(f"Drug network loaded: {drug_net.number_of_nodes()} nodes, {drug_net.number_of_edges()} edges.")
    # --- Centrality Measures Calculation ---
    # 1. Weighted Degree (Strength)
    print("\nCalculating Weighted Degree (Strength)...")
    start_time = time.time()
    strength = dict(drug_net.degree(weight='weight'))
    end_time = time.time()
    print(f"Strength calculation complete. (Took {end_time - start_time:.2f} seconds)")

    # 2. Betweenness Centrality
    # NOTE: Calculating weighted betweenness centrality on a graph this large can be
    # extremely time-consuming (hours or days). We'll calculate the unweighted version
    # for speed as a good approximation for this project scope. Add k= parameter to speed up.
    # Set k to sample ~10% of nodes for approximation if needed: k = drug_net.number_of_nodes() // 10
    print("\nCalculating Betweenness Centrality (Unweighted Approximation)...")
    start_time = time.time()
    # Use k=None to calculate exact betweenness, remove if too slow. Try k=1000 first.
    betweenness = nx.betweenness_centrality(drug_net, k=1000, normalized=True, weight=None)
    end_time = time.time()
    print(f"Betweenness calculation complete. (Took {end_time - start_time:.2f} seconds)")

    # 3. Eigenvector Centrality
    # Weighted eigenvector centrality can also be slow and sometimes fails to converge.
    # Let's use the unweighted version for robustness.
    print("\nCalculating Eigenvector Centrality (Unweighted)...")
    start_time = time.time()
    try:
        # Increased max_iter for potentially better convergence on large graphs
        eigenvector = nx.eigenvector_centrality(drug_net, max_iter=1000, tol=1e-04)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge. Try increasing max_iter or using alternative methods.")
        eigenvector = {node: 0.0 for node in drug_net.nodes()} # Assign default value if fails
    end_time = time.time()
    print(f"Eigenvector calculation complete. (Took {end_time - start_time:.2f} seconds)")

    # --- Combine Results into a DataFrame ---
    centrality_df = pd.DataFrame({
        'DrugName': list(drug_net.nodes()),
        'Strength': [strength.get(node, 0) for node in drug_net.nodes()],
        'Betweenness': [betweenness.get(node, 0.0) for node in drug_net.nodes()],
        'Eigenvector': [eigenvector.get(node, 0.0) for node in drug_net.nodes()]
    })

    # --- Display Top Ranked Drugs ---
    print("\n--- Top 10 Drugs by Strength ---")
    print(centrality_df.sort_values(by='Strength', ascending=False).head(10))
    
    print("\n--- Top 10 Drugs by Betweenness Centrality ---")
    print(centrality_df.sort_values(by='Betweenness', ascending=False).head(10))
    
    print("\n--- Top 10 Drugs by Eigenvector Centrality ---")
    print(centrality_df.sort_values(by='Eigenvector', ascending=False).head(10))
    
    # --- Save Centrality Results ---
    results_file = BASE_DIR / 'results' / 'drug_centrality_scores.csv'
    centrality_df.to_csv(results_file, index=False)
    print(f"\nCentrality scores saved to '{results_file}'")