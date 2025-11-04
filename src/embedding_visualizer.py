# Part of Phase 3: Embedding Visualization
import pickle
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap # Use umap-learn

# Define BASE_DIR relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent

def visualize_embeddings_umap(embeddings_file_path, communities_file_path, plot_output_path):
    """
    Loads embeddings and communities, performs UMAP, and creates a visualization.

    Args:
        embeddings_file_path (pathlib.Path): Path to the node embeddings file (.pkl).
        communities_file_path (pathlib.Path): Path to the community partition file (.pkl).
        plot_output_path (pathlib.Path): Path to save the output plot (.png).
    """
    print(f"\n--- Starting Embedding Visualization ---")
    print("Loading communities and embeddings...")

    try:
        with open(communities_file_path, 'rb') as f:
            # This contains {node: community_id} for ALL original nodes
            partition = pickle.load(f)
        with open(embeddings_file_path, 'rb') as f:
            # This contains {node: embedding_vector} ONLY for nodes in the pruned graph
            embeddings = pickle.load(f)
        print("Embeddings and communities loaded.")
    except FileNotFoundError as e:
        print(f"Error: Could not load required files: {e}")
        return
    except Exception as e:
        print(f"An error occurred loading files: {e}")
        return

    # --- Prepare data for UMAP ---
    # Important: Only use nodes that exist in BOTH the embeddings and the partition
    
    # Get the list of nodes that were actually embedded
    embedded_nodes = list(embeddings.keys())
    if not embedded_nodes:
        print("Error: No embeddings found in the file.")
        return
        
    embedding_vectors = np.array([embeddings[node] for node in embedded_nodes])
    # Get community IDs ONLY for the nodes that were embedded
    community_ids = [partition.get(node, -1) for node in embedded_nodes] # Use -1 for nodes missing partition info (shouldn't happen if partition is complete)

    print(f"Preparing data for {len(embedded_nodes)} nodes with embeddings.")

    # --- Run UMAP ---
    print("Running UMAP for dimensionality reduction...")
    start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, # Standard UMAP parameter
                          min_dist=0.1,  # Standard UMAP parameter
                          n_components=2, # Reduce to 2D
                          metric='cosine', # Good metric for high-dimensional vectors
                          random_state=42) # For reproducibility
    embedding_2d = reducer.fit_transform(embedding_vectors)
    end_time = time.time()
    print(f"UMAP complete. (Took {end_time - start_time:.2f} seconds)")


    # --- Create Visualization Plot ---
    print("\nCreating visualization plot...")
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'community': community_ids,
        'node': embedded_nodes # Keep track of node names if needed later
    })

    # Find the largest communities among the embedded nodes
    unique_communities = sorted(list(set(community_ids)))
    num_unique_communities = len(unique_communities)
    print(f"Number of unique communities represented in embeddings: {num_unique_communities}")

    # Determine top communities based on embedded nodes
    community_counts = plot_df['community'].value_counts()
    top_communities = community_counts.nlargest(10).index # Show top 10 communities
    plot_df_filtered = plot_df[plot_df['community'].isin(top_communities)]
    print(f"Plotting the {len(top_communities)} largest communities found among embedded nodes.")


    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=plot_df_filtered,
        x='x',
        y='y',
        hue='community',
        palette=sns.color_palette("tab10", n_colors=len(top_communities)), # Colors for top N communities
        s=30,      # Point size
        alpha=0.7, # Point transparency
        legend='full' # Show legend
    )
    plt.title('2D UMAP Visualization of Disease Embeddings by Community', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Community ID', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Save the plot
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"\n--- Embedding Visualization Complete! ---")
    print(f"Plot saved to '{plot_output_path}'")
    # plt.show() # Uncomment to display plot
