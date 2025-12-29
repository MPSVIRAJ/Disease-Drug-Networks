import pickle
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap 


BASE_DIR = Path(__file__).resolve().parent.parent

def visualize_embeddings_umap(embeddings_file_path, communities_file_path, plot_output_path):
    """
    Visualizes high-dimensional node embeddings using UMAP dimensionality reduction.

    This function loads pre-computed node embeddings and community partitions, 
    aligns the data, and projects the high-dimensional vectors into 2D space using 
    UMAP (Uniform Manifold Approximation and Projection). The resulting scatter 
    plot is color-coded by community assignment, displaying only the top 10 largest 
    communities to ensure visual clarity.

    Args:
        embeddings_file_path (str or Path): Path to the pickle file containing 
            the dictionary of node embeddings {node: vector}.
        communities_file_path (str or Path): Path to the pickle file containing 
            the dictionary of community assignments {node: community_id}.
        plot_output_path (str or Path): The destination path where the generated 
            2D visualization plot will be saved.

    Returns:
        None: The function saves a high-resolution PNG plot to disk.
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

    # Prepare data for UMAP
    # Only use nodes that exist in BOTH the embeddings and the partition
    
    # Get the list of nodes that were actually embedded
    embedded_nodes = list(embeddings.keys())
    if not embedded_nodes:
        print("Error: No embeddings found in the file.")
        return
        
    embedding_vectors = np.array([embeddings[node] for node in embedded_nodes])
    # Get community IDs ONLY for the nodes embedded
    community_ids = [partition.get(node, -1) for node in embedded_nodes] # -1 for nodes without a community

    print(f"Preparing data for {len(embedded_nodes)} nodes with embeddings.")

    # Run UMAP 
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


    # Create Visualization Plot 
    print("\nCreating visualization plot...")
    plot_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'community': community_ids,
        'node': embedded_nodes 
    })

    # Find the largest communities among the embedded nodes
    unique_communities = sorted(list(set(community_ids)))
    num_unique_communities = len(unique_communities)
    print(f"Number of unique communities represented in embeddings: {num_unique_communities}")

    # Determine top communities based on embedded nodes
    community_counts = plot_df['community'].value_counts()
    top_communities = community_counts.nlargest(10).index 
    plot_df_filtered = plot_df[plot_df['community'].isin(top_communities)]
    print(f"Plotting the {len(top_communities)} largest communities found among embedded nodes.")

    sns.set_context("talk", font_scale=1.1)  
    plt.style.use('seaborn-v0_8-whitegrid')  

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=plot_df_filtered,
        x='x',
        y='y',
        hue='community',
        palette=sns.color_palette("tab10", n_colors=len(top_communities)), 
        s=60,      
        alpha=0.7, 
        legend='full' 
    )
    plt.title('2D UMAP Visualization of Disease Embeddings by Community', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=16, labelpad=15)
    plt.ylabel('UMAP Dimension 2', fontsize=16, labelpad=15)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(title='Community ID', title_fontsize=16, fontsize=14, 
               bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    plt.grid(False) 
    plt.grid(visible=False, which='both', axis='both')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"\n--- Embedding Visualization Complete! ---")
    print(f"Plot saved to '{plot_output_path}'")

