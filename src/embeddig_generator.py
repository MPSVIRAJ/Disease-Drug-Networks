# src/embedding_generator.py
import networkx as nx
import pickle
from pathlib import Path
import time
# --- Use nodevectors ---
from nodevectors import Node2Vec
import pandas as pd
import numpy as np # Needed for saving embeddings
from scipy.sparse import csr_matrix

def generate_node_embeddings_fast(pruned_network_path, embeddings_output_path):
    """
    Loads the PRUNED disease network, generates embeddings using nodevectors,
    and saves them. Handles NetworkX 3.x compatibility.

    Args:
        pruned_network_path (pathlib.Path): Path to the PRUNED network file (.pkl).
        embeddings_output_path (pathlib.Path): Path to save the output embeddings (.pkl).
    """

    print(f"\n--- Starting Phase 3: Node Embedding (Fast Version) ---")
    print(f"Loading PRUNED disease network from '{pruned_network_path}'...")

    # ...(loading code remains the same)...
    try:
        with open(pruned_network_path, 'rb') as f:
            disease_net_pruned = pickle.load(f)
        if disease_net_pruned is None:
             raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: Pruned network file not found or is invalid at {pruned_network_path}")
        return

    print(f"Pruned disease network loaded: {disease_net_pruned.number_of_nodes()} nodes, {disease_net_pruned.number_of_edges()} edges.")

    # --- FIX: Convert graph to sparse matrix BEFORE fitting ---
    print("\nConverting graph to SciPy sparse matrix for nodevectors...")
    # Ensure nodes are in a fixed order for the matrix
    node_list = list(disease_net_pruned.nodes())
    # Use the correct NetworkX 3.x function
    adjacency_matrix_sparse = nx.adjacency_matrix(disease_net_pruned, nodelist=node_list, weight='weight')
    # Ensure it's in CSR format, which nodevectors/csrgraph expects
    if not isinstance(adjacency_matrix_sparse, csr_matrix):
         adjacency_matrix_sparse = csr_matrix(adjacency_matrix_sparse)
    print("Conversion complete.")


    # --- Configure and Run nodevectors Node2Vec ---
    print("\nConfiguring and running nodevectors Node2Vec...")
    start_time = time.time()

    g2v = Node2Vec(
        n_components=64,
        walklen=10,
        epochs=10,
        return_weight=1.0,
        neighbor_weight=1.0,
        threads=4,
        keep_walks=False,
        verbose=True
    )

    # --- FIX: Fit using the pre-converted sparse matrix ---
    # The fit method in nodevectors should accept a csr_matrix
    g2v.fit(adjacency_matrix_sparse)

    end_time = time.time()
    print(f"Nodevectors training complete! (Took {end_time - start_time:.2f} seconds)")

    # --- Save the Embeddings ---
    print(f"\nSaving node embeddings...")

    # --- FIX: Get embeddings using the original node names from node_list ---
    # g2v.predict() works with integer indices when fit on a matrix. Map back to names.
    embedding_dict = {}
    for i, node_name in enumerate(node_list):
        embedding_dict[node_name] = g2v.predict(i) # Use index i

    with open(embeddings_output_path, 'wb') as f:
        pickle.dump(embedding_dict, f)

    print(f"Node embeddings saved to '{embeddings_output_path}'")
    print(f"  - Embedding dimension: {g2v.n_components}")
    print(f"  - Number of nodes embedded: {len(embedding_dict)}")