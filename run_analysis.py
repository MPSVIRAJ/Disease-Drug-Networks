# run_analysis.py
from pathlib import Path
import argparse
from src.data_loader import load_and_clean_data
from src.network_builder import build_and_project_networks, prune_network_by_weight
from src.desease_community_analysis import analyze_disease_network
from src.drug_analysis import analyze_drug_network
from src.embeddig_generator import generate_node_embeddings_fast
from src.embedding_visualizer import visualize_embeddings_umap

#diffusion model modules
from src.data_splitter import create_validation_split
from src.diffusion_model import run_diffusion
from src.model_evaluator import evaluate_predictions

# Define the absolute path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent

data_file = BASE_DIR / 'data' / 'CTD_chemicals_diseases.csv'
network_output_dir = BASE_DIR / 'results' / 'networks'
plot_output_dir = BASE_DIR / 'results' / 'plots'

# --- Network file paths ---
disease_net_path = network_output_dir/ "disease_network.pkl"
drug_net_path = network_output_dir / "drug_network.pkl"

# --- Embedding output paths ---
disease_net_pruned_path = network_output_dir / "disease_network_pruned_p90.pkl" 
disease_embeddings_path = network_output_dir / "disease_embeddings_fast.pkl"
disease_communities_path = network_output_dir / "disease_communities.pkl"
embedding_plot_path = plot_output_dir / "disease_embedding_umap.png"

def main(args):
    # --- Create directories for results if they don't exist ---
    network_output_dir.mkdir(parents=True, exist_ok=True)
    plot_output_dir.mkdir(parents=True, exist_ok=True)    
    edge_list = None

    run_phase1 = args.phase1 or args.all or \
                 ((args.phase2 or args.phase3) and not (disease_net_path.exists() and drug_net_path.exists()))
    
    if run_phase1 or args.phase4 or args.all:
        if not data_file.exists():
            print(f"Error: Data file not found at {data_file}")
            return
        edge_list = load_and_clean_data(data_file)
        if edge_list is None:
            print("Error: Failed to load data.")
            return
    
    # --- Phase 1: Data Loading & Network Construction ---
    if run_phase1:
        print("\n--- Running Phase 1: Data Loading & Network Construction ---")
        edge_list = load_and_clean_data(data_file)
        build_and_project_networks(edge_list, network_output_dir)
        print("\n--- Workflow Phase 1 Complete ---")
    elif (args.phase2 or args.phase3):
         print("\nSkipping Phase 1: Found existing projected networks.")


    # --- Phase 2: Core Analysis ---
    if args.phase2 or args.all:
        print("\n--- Running Phase 2: Core Analysis ---")
        if disease_net_path.exists():
             analyze_disease_network(disease_net_path)
        else:
             print("Error: Disease network file not found for community analysis. Run Phase 1 first.")

        if drug_net_path.exists():
            analyze_drug_network(drug_net_path)
        else:
            print("Error: Drug network file not found for centrality analysis. Run Phase 1 first.")
        print("\n--- Workflow Phase 2 Complete ---")
    else:
        print("\nSkipping Phase 2.")


# --- Phase 3: Embedding Generation & Visualization ---
    if args.phase3 or args.all:
        print("\n--- Running Phase 3: Advanced Validation ---")
        
        # Check for prerequisites
        if not disease_net_path.exists():
            print("Error: Original disease network file not found. Run Phase 1 first.")
            return
        if not disease_communities_path.exists():
            print("Error: Community file not found. Run Phase 2 first.")
            return

        # --- Embedding Generation ---
        if not disease_embeddings_path.exists():
            print(f"Embeddings file not found. Generating embeddings...")
            
            # Pruning step (ensure pruned file exists or create it)
            if not disease_net_pruned_path.exists():
                 print("Pruned network not found. Generating pruned network first...")
                 pruned_graph = prune_network_by_weight(disease_net_path, disease_net_pruned_path, percentile=90)
                 if not pruned_graph:
                     print("Error during pruning. Cannot proceed with embedding.")
                     return
            else:
                 print("Found existing pruned network.")
            
            # Generate embeddings
            generate_node_embeddings_fast(disease_net_pruned_path, disease_embeddings_path)
        else:
            print("Found existing embeddings file. Skipping generation.")

        # --- Visualization ---
        print("\nProceeding with visualization...")
        visualize_embeddings_umap(disease_embeddings_path, disease_communities_path, embedding_plot_path)
        print("\n--- Workflow Phase 3 Complete ---")
    else:
        print("\nSkipping Phase 3.")
    
    if args.phase4 or args.all:
        print("\n--- Running Phase 4: Diffusion Method ---")
        if edge_list is None: 
            print("Error: Edge list not loaded for Phase 4.")
            return
            
        all_drugs = edge_list['ChemicalName'].unique()
        all_diseases = edge_list['DiseaseName'].unique()
        
        # Create the split
        train_df, test_df = create_validation_split(edge_list, test_fraction=0.1)
        
        # Run the diffusion
        Final_Scores, drug_to_idx, disease_to_idx = run_diffusion(train_df, all_drugs, all_diseases)
        
        # Evaluate the results
        auc, auprc = evaluate_predictions(Final_Scores, test_df, drug_to_idx, disease_to_idx)
        
        # Generate the Plot immediately
        from src.model_evaluator import plot_score_distribution
        plot_score_distribution()
        
        with open(network_output_dir / "diffusion_metrics.txt", "w") as f:
            f.write(f"AUROC: {auc:.4f}\nAUPRC: {auprc:.4f}\n")
        print("\n--- Workflow Phase 4 Complete ---")
    else:
        print("\nSkipping Phase 4.")
    

if __name__ == "__main__":
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Run phases of the Disease-Drug Network analysis.")
    parser.add_argument('--phase1', action='store_true', help="Run Phase 1: Data Loading & Network Construction.")
    parser.add_argument('--phase2', action='store_true', help="Run Phase 2: Core Analysis (Communities & Centrality).")
    parser.add_argument('--phase3', action='store_true', help="Run Phase 3: Advanced Validation (Embedding & Visualization).")
    parser.add_argument('--phase4', action='store_true', help="Run Phase 4: Diffusion/Validation Method.")
    parser.add_argument('--all', action='store_true', help="Run all phases.")

    parsed_args = parser.parse_args()
    # If no args given, print help
    if not any(vars(parsed_args).values()):
         parser.print_help()
    else:
        main(parsed_args)