# run_analysis.py
from pathlib import Path
import argparse
import pandas as pd 


from src.data_loader import load_and_clean_data
from src.network_builder import build_and_project_networks, prune_network_by_weight
from src.desease_community_analysis import analyze_disease_network
from src.drug_analysis import analyze_drug_network
from src.embeddig_generator import generate_node_embeddings_fast
from src.embedding_visualizer import visualize_embeddings_umap, visualize_spring_layout

#diffusion model modules
from src.diffusion_model import run_diffusion_model
from src.model_evaluator import evaluate_predictions, plot_score_distribution

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
spring_plot_path = plot_output_dir / "disease_spring_layout.png"


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
        
        # --- Disease Network Loop ---
        if disease_net_path.exists():
            disease_stats = []
            print("\n>>> Analyzing Disease Network Sensitivities...")
            
            # Define resolutions to test (Sensitivity Analysis)
            # 1.0 = Standard Louvain. 1.2, 1.5 = Force smaller, more granular communities
            resolutions_to_test = [1.0, 1.2, 1.5] 
            
            for res in resolutions_to_test:
                # analyze_disease_network now returns a DataFrame
                df_res = analyze_disease_network(disease_net_path, resolution=res)
                if df_res is not None:
                    disease_stats.append(df_res)
            
            # Combine all results into one table and save
            if disease_stats:
                final_dis_df = pd.concat(disease_stats, ignore_index=True)
                 
                # Create tables directory if it doesn't exist
                tables_dir = network_output_dir.parent / "tables"
                tables_dir.mkdir(parents=True, exist_ok=True)
                 
                csv_path = tables_dir / "disease_community_stats.csv"
                final_dis_df.to_csv(csv_path, index=False)
                print(f"SUCCESS: Aggregated disease stats saved to {csv_path}")
        else:
            print("Error: Disease network file not found. Run Phase 1 first.")
        
        # --- Drug Network---
        """
        if drug_net_path.exists():
            analyze_drug_network(drug_net_path)
        else:
            print("Error: Drug network file not found for centrality analysis. Run Phase 1 first.")
        print("\n--- Workflow Phase 2 Complete ---")
        """
    else:
        print("\nSkipping Phase 2.")

# --- Phase 3: Embedding Generation & Visualization ---
    if args.phase3 or args.all:
        print("\n--- Running Phase 3: Visualization ---")
        
        # Select the Best Community Partition (from Phase 2)
        target_comm_file = network_output_dir / "disease_communities_res1p5.pkl"
        if not target_comm_file.exists():
             avail_files = list(network_output_dir.glob("disease_communities*.pkl"))
             if avail_files:
                 target_comm_file = avail_files[0]
                 print(f"Target res 1.5 missing. Using fallback: {target_comm_file.name}")
             else:
                 print("Error: No community file found. Run Phase 2 first.")
                 return
        else:
             print(f"Using community partition: {target_comm_file.name}")

        # Prune the Network (Crucial for Speed/Clarity)
        if not disease_net_pruned_path.exists():
             print("Pruning disease network for clean visualization...")
             prune_network_by_weight(disease_net_path, disease_net_pruned_path, percentile=90)
        
        # Generate Embeddings (for UMAP)
        if not disease_embeddings_path.exists():
            print("Generating node embeddings...")
            generate_node_embeddings_fast(disease_net_pruned_path, disease_embeddings_path)
       
        # Generate Plots
        print("Generating plots...")
        
        # Plot A: UMAP
        visualize_embeddings_umap(disease_embeddings_path, target_comm_file, embedding_plot_path)
        
        # Plot B: Spring Layout
        visualize_spring_layout(disease_net_pruned_path, target_comm_file, spring_plot_path)
        
        print("\n--- Workflow Phase 3 Complete ---")
    else:
        print("\nSkipping Phase 3.")

# --- Phase 4: Diffusion/Validation Method ---
    if args.phase4 or args.all:
        print("\n--- Running Phase 4: Diffusion Model Validation ---")
        
        # Prepare Data
        # Prefer using the data already loaded in memory (edge_list)
        # instead of failing if a temp CSV file is missing.
        drug_disease_df = None
        
        if 'edge_list' in locals() and edge_list is not None:
            print("Using data loaded from memory...")
            # Ensure it is a DataFrame with correct columns
            if isinstance(edge_list, pd.DataFrame):
                drug_disease_df = edge_list
            else:
                # If edge_list is a list of tuples, convert it
                drug_disease_df = pd.DataFrame(edge_list, columns=['ChemicalName', 'DiseaseName'])
        
        # Fallback: If edge_list was missing for some reason, try loading CSV
        elif data_file.exists():
            print(f"Loading data from file: {data_file}")
            drug_disease_df = pd.read_csv(data_file)
        
        # Run Analysis if Data Found
        if drug_disease_df is not None:
            # Ensure correct column names for the model
            # (Standardize to 'ChemicalName' and 'DiseaseName' if they are 0 and 1)
            if 'ChemicalName' not in drug_disease_df.columns:
                drug_disease_df.columns = ['ChemicalName', 'DiseaseName']

            print(f"Data ready. Total interactions: {len(drug_disease_df)}")
            
            # Create simple node lists
            drug_df = pd.DataFrame({'ChemicalName': drug_disease_df['ChemicalName'].unique()})
            disease_df = pd.DataFrame({'DiseaseName': drug_disease_df['DiseaseName'].unique()})

            # Run Diffusion (using the wrapper that handles splitting)
            # This returns: Scores, Test Set, Mappings, and Training Set
            Final_Scores, test_df, drug_to_idx, disease_to_idx, train_df = run_diffusion_model(
                drug_disease_df, drug_df, disease_df, alpha=0.5
            )
            
            # Evaluate (Train + test metrics)

            evaluate_predictions(Final_Scores, test_df, drug_to_idx, disease_to_idx, train_df=train_df)
            
            # Plot
            plot_score_distribution()
            
            print("\n--- Workflow Phase 4 Complete ---")
        else:
            print("Error: Could not load interaction data.")
            print("Ensure 'CTD_chemicals_diseases.csv' is in your data/ folder.")
    

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