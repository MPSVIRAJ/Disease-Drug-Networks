import pickle
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap 
import networkx as nx


BASE_DIR = Path(__file__).resolve().parent.parent

def visualize_embeddings_umap(embeddings_file_path, communities_file_path, plot_output_path):
    """
    Projects node embeddings into 2D with UMAP and visualizes communities (top 10 by size).

    This function creates a publication-quality embedding visualization by:
    1. Loading a node->community partition mapping from `communities_file_path`.
    2. Loading precomputed node embeddings from `embeddings_file_path`.
    3. Aligning nodes across both sources (only nodes that have embeddings are plotted,
       and their community IDs are looked up from the partition; missing IDs use -1).
    4. Running UMAP to reduce the high-dimensional embedding vectors to 2D coordinates
       using a cosine metric and a fixed random seed for reproducibility.
    5. Building a plotting DataFrame containing coordinates, node names, and community IDs.
    6. Selecting only the 10 largest communities represented among the embedded nodes
       to keep the scatter plot readable.
    7. Creating and saving a labeled Seaborn scatter plot with points colored by community.

    Args:
        embeddings_file_path (str or Path): Path to a pickle file containing embeddings
            as a dictionary {node: embedding_vector}. Typically only nodes from the pruned
            graph are embedded.
        communities_file_path (str or Path): Path to a pickle file containing community
            assignments as a dictionary {node: community_id} for the full (original) graph.
        plot_output_path (str or Path): Output path (including filename) where the PNG
            visualization will be saved.

    Notes:
        - Uses UMAP parameters: n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine',
          random_state=42 for stable and comparable plots.
        - Communities are filtered based on their frequency among the *embedded* nodes,
          not necessarily the full graph.
        - Nodes missing from the partition get community=-1; these are typically excluded
          by the “top 10 communities” filtering.

    Returns:
        None: Saves a high-resolution PNG plot to `plot_output_path`.
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


def visualize_spring_layout(network_file_path, communities_file_path, output_path):
    """
    Visualizes a (pruned) disease network using a force-directed Spring layout.

    This function generates a readable network visualization by:
    1. Loading a pickled NetworkX graph from `network_file_path` (intended to be a pruned
       graph for speed and reduced clutter).
    2. Optionally loading a Louvain partition mapping from `communities_file_path` and
       assigning node colors by community ID (nodes missing from the partition default to 0).
       If no community file is found, all nodes are drawn in a single default color.
    3. Computing a reproducible Spring layout (force-directed) using a fixed seed and
       tuned parameters (k=0.15, iterations=50) to balance spacing and runtime.
    4. Drawing nodes (colored by community if available) and faint edges for context.
    5. Saving a high-resolution PNG image to `output_path`.

    Args:
        network_file_path (str or Path): Path to the pickled NetworkX graph used for plotting.
            This is typically the pruned disease network (e.g., p90 threshold) to improve
            performance and visual clarity.
        communities_file_path (str or Path): Path to a pickled partition dictionary
            {node: community_id}. If missing, the plot is generated without community coloring.
        output_path (str or Path): Destination path (including filename) where the plot
            will be saved.

    Notes:
        - Uses the 'tab20' colormap for community coloring, suitable for moderate numbers
          of communities (colors will repeat visually if community IDs exceed the palette size).
        - Edges are drawn with low alpha to reduce clutter in dense graphs.

    Returns:
        None: Saves the figure to `output_path`.
    """
    print(f"\n--- Generating Spring Layout Visualization ---")
    
    if not Path(network_file_path).exists():
        print(f"Error: Network file not found at {network_file_path}")
        return

    # Load Graph
    with open(network_file_path, 'rb') as f:
        G = pickle.load(f)
    
    # Load Communities (if available)
    node_colors = []
    if Path(communities_file_path).exists():
        with open(communities_file_path, 'rb') as f:
            partition = pickle.load(f)
        # Map colors (default to 0 if node missing from partition)
        node_colors = [partition.get(node, 0) for node in G.nodes()]
    else:
        node_colors = 'skyblue' # Default color if no community file
        
    print(f"Graph loaded: {len(G)} nodes. Running Spring Layout...")
    print("Optimization: Using sparse iterations (k=0.15, iter=50).")
    
    # Spring Layout Calculation
    # k=0.15: Increases spacing between nodes
    # iterations=50: Enough to settle the layout without taking forever
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    plt.figure(figsize=(12, 12))
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.tab20, 
                           node_color=node_colors, alpha=0.8)
    
    # Draw Edges (Faint)
    nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='gray')
    
    plt.title("Disease Network - Spring Layout (Pruned)", fontsize=20)
    plt.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spring layout saved to '{output_path}'")
