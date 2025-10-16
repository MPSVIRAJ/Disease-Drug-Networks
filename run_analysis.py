# run_analysis.py
from pathlib import Path
from src.data_loader import load_and_clean_data
from src.network_builder import build_and_project_networks

# Define the absolute path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent

def main():
    """Main script to run the network analysis workflow."""
    
    # --- Define OS-independent paths ---
    data_file = BASE_DIR / 'data' / 'CTD_chemicals_diseases.csv'
    network_output_dir = BASE_DIR / 'results' / 'networks'
    plot_output_dir = BASE_DIR / 'results' / 'plots'
    
    # --- Create directories for results if they don't exist ---
    network_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Phase 1: Data Loading and Network Construction ---
    edge_list = load_and_clean_data(data_file)
    disease_net, drug_net = build_and_project_networks(edge_list, network_output_dir)
    
    print("\n--- Workflow Phase 1 Complete ---")
    print(f"Disease network has {disease_net.number_of_nodes():,} nodes and {disease_net.number_of_edges():,} edges.")
    print(f"Drug network has {drug_net.number_of_nodes():,} nodes and {drug_net.number_of_edges():,} edges.")

if __name__ == "__main__":
    main()