# run_analysis.py
import os
from src.data_loader import load_and_clean_data
from src.network_builder import build_and_project_networks

def main():
    """Main script to run the network analysis workflow."""
    
    # Define file paths
    data_file = 'data/CTD_chemicals_diseases.csv'
    
    # Create directories for results if they don't exist
    os.makedirs('results/networks', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # --- Phase 1: Data Loading and Network Construction ---
    edge_list = load_and_clean_data(data_file)
    disease_net, drug_net = build_and_project_networks(edge_list)
    
    print("\n--- Workflow Phase 1 Complete ---")
    print(f"Disease network has {disease_net.number_of_nodes()} nodes and {disease_net.number_of_edges()} edges.")
    print(f"Drug network has {drug_net.number_of_nodes()} nodes and {drug_net.number_of_edges()} edges.")

if __name__ == "__main__":
    main()