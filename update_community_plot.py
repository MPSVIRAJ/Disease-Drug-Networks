import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
COMMUNITY_FILE = BASE_DIR / 'results' / 'networks' / 'disease_communities.pkl'
PLOT_OUTPUT = BASE_DIR / 'results' / 'plots' / 'community_size_distribution.png'

def regenerate_plot():
    """
    Regenerates the Community Size Distribution plot with improved styling.

    This utility function loads existing community partition data (saved during 
    Phase 2) and renders a high-resolution log-scale histogram using updated 
    Seaborn aesthetics (larger font sizes, cleaner grids). This allows for 
    plot tweaks without re-running the computationally expensive Louvain algorithm.

    Returns:
        None: Overwrites the plot at 'results/plots/community_size_distribution.png'.
    """
    
    print(f"Loading communities from {COMMUNITY_FILE}...")
    
    if not COMMUNITY_FILE.exists():
        print("Error: Community file not found. You must run Phase 2 at least once.")
        return

    # Load the existing partition
    with open(COMMUNITY_FILE, 'rb') as f:
        partition = pickle.load(f)
    
    # Process Data (Count sizes)
    community_sizes = {}
    for node, comm_id in partition.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
    size_df = pd.DataFrame(community_sizes.items(), columns=['CommunityID', 'Size'])
    num_communities = len(size_df)
    
    print(f"Loaded {len(partition)} nodes across {num_communities} communities.")

    print("Generating plot with larger text...")
    
    sns.set_context("talk", font_scale=1.1)
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 8))
    plt.hist(size_df['Size'], bins=50, log=True, color='#4c72b0', edgecolor='black', alpha=0.8)
    plt.grid(False) 
    plt.grid(visible=False, which='both', axis='both')
    plt.title('Distribution of Disease Community Sizes', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Community Size (Number of Diseases)', fontsize=18, labelpad=15)
    plt.ylabel('Frequency (Log Scale)', fontsize=18, labelpad=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    sns.despine()
    plt.tight_layout()
    
    # Save
    plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"Success! Updated plot saved to: {PLOT_OUTPUT}")

if __name__ == "__main__":
    regenerate_plot()