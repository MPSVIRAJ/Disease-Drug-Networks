# run_analysis.py
from pathlib import Path
import argparse
from src.data_loader import load_and_clean_data
from src.network_builder import build_and_project_networks, prune_network_by_weight
from src.desease_community_analysis import analyze_disease_network
from src.drug_analysis import analyze_drug_network
from src.embeddig_generator import generate_node_embeddings_fast
from src.embedding_visualizer import visualize_embeddings_umap

# Define the absolute path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent

data_file = BASE_DIR / 'data' / 'CTD_chemicals_diseases.csv'
network_output_dir = BASE_DIR / 'results' / 'networks'
plot_output_dir = BASE_DIR / 'results' / 'plots'

# --- Network file paths ---
disease_net_path = network_output_dir/ "disease_network.pkl"
drug_net_path = network_output_dir / "drug_network.pkl"

# --- Embedding output paths ---
disease_net_pruned_path = network_output_dir / "disease_network_pruned_p90.pkl" # p90 means kept top 10%
disease_embeddings_path = network_output_dir / "disease_embeddings_fast.pkl"
disease_communities_path = network_output_dir / "disease_communities.pkl"
embedding_plot_path = plot_output_dir / "disease_embedding_umap.png"

def main(args):
    # --- Create directories for results if they don't exist ---
    network_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)    
    
    run_phase1 = args.phase1 or args.all or \
                 ((args.phase2 or args.phase3) and not (disease_net_path.exists() and drug_net_path.exists()))
    if run_phase1:
        print("\n--- Running Phase 1: Data Loading & Network Construction ---")
        edge_list = load_and_clean_data(data_file)
        build_and_project_networks(edge_list, network_output_dir)
        print("\n--- Workflow Phase 1 Complete ---")
        pass
    elif (args.phase2 or args.phase3):
         print("\nSkipping Phase 1: Found existing projected networks.")


    # --- Phase 2: Core Analysis ---
    if args.phase2 or args.all:
        print("\n--- Running Phase 2: Core Analysis ---")
        if disease_net_path.exists():
             analyze_disease_network()
        else:
             print("Error: Disease network file not found for community analysis. Run Phase 1 first.")

        if drug_net_path.exists():
            analyze_drug_network(drug_net_path)
        else:
            print("Error: Drug network file not found for centrality analysis. Run Phase 1 first.")
        print("\n--- Workflow Phase 2 Complete ---")
        pass
    else:
        print("\nSkipping Phase 2.")


# --- Phase 3: Embedding Generation & Visualization ---
    if args.phase3 or args.all:
        print("\n--- Running Phase 3: Advanced Validation ---")

        # --- Embedding Generation (Run only if needed) ---
        if not disease_embeddings_path.exists():
            print(f"Embeddings file not found at '{disease_embeddings_path}'. Generating embeddings...")
            if disease_net_path.exists():
                # Pruning step (ensure pruned file exists or create it)
                if not disease_net_pruned_path.exists():
                     print("Pruned network not found. Generating pruned network first...")
                     pruned_graph = prune_network_by_weight(disease_net_path, disease_net_pruned_path, percentile=90)
                     if not pruned_graph:
                         print("Error during pruning. Cannot proceed with embedding.")
                         return # Exit if pruning fails
                else:
                     print("Found existing pruned network.")

                # Generate embeddings using the pruned network
                if disease_net_pruned_path.exists():
                     generate_node_embeddings_fast(disease_net_pruned_path, disease_embeddings_path)
                else:
                     # This case should ideally not happen if pruning worked
                     print("Error: Pruned network file missing after attempting creation.")
                     return
            else:
                print("Error: Original disease network file not found for pruning/embedding. Run Phase 1 first.")
                return # Exit if original network is missing
        else:
            print(f"Found existing embeddings file at '{disease_embeddings_path}'. Skipping generation.")

        # --- Visualization (Run if embeddings and communities exist) ---
        if disease_embeddings_path.exists() and disease_communities_path.exists():
            print("\nProceeding with visualization...")
            visualize_embeddings_umap(disease_embeddings_path, disease_communities_path, embedding_plot_path)
            print("\n--- Workflow Phase 3 Complete ---")
        elif not disease_communities_path.exists():
             print("Error: Community file not found. Run Phase 2 first to generate communities.")
        else:
             print("Error: Embeddings file still not found after attempting generation.")

    else:
        print("\nSkipping Phase 3.")


if __name__ == "__main__":
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Run phases of the Disease-Drug Network analysis.")
    parser.add_argument('--phase1', action='store_true', help="Run Phase 1: Data Loading & Network Construction.")
    parser.add_argument('--phase2', action='store_true', help="Run Phase 2: Core Analysis (Communities & Centrality).")
    parser.add_argument('--phase3', action='store_true', help="Run Phase 3: Advanced Validation (Embedding & Visualization).")
    parser.add_argument('--all', action='store_true', help="Run all phases.")

    # If no arguments are given, default to running all phases? Or maybe just print help?
    # Let's default to printing help if no specific phase is chosen.
    parsed_args = parser.parse_args()

    # Default to running all if no specific phase is selected
    if not any([parsed_args.phase1, parsed_args.phase2, parsed_args.phase3, parsed_args.all]):
         # print("No phase selected. Running all phases by default.")
         # parsed_args.all = True
         # OR print help:
         parser.print_help()
    else:
        main(parsed_args) # Pass parsed arguments to main