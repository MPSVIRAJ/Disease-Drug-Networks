# src/model_evaluator.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results' / 'networks'
PLOT_DIR = BASE_DIR / 'results' / 'plots'
DATA_FILE = RESULTS_DIR / 'diffusion_results.npz'

def evaluate_predictions(Scores_matrix, test_df, drug_to_idx, disease_to_idx):
    """
    Evaluates the performance of the diffusion model using AUC and AUPRC metrics.

    This function extracts prediction scores for the test set pairs (both positive 
    and negative) from the pre-computed score matrix. It compares these scores 
    against the true labels to calculate the Area Under the ROC Curve (AUROC) and 
    Area Under the Precision-Recall Curve (AUPRC). 
    
    The raw scores and labels are also saved to an `.npz` file for later analysis 
    or plotting.

    Args:
        Scores_matrix (scipy.sparse.csr_matrix or np.ndarray): The matrix of predicted 
            association scores (Diseases x Drugs).
        test_df (pd.DataFrame): The test dataset containing 'ChemicalName', 
            'DiseaseName', and 'Label' (1 for positive, 0 for negative).
        drug_to_idx (dict): Mapping of drug names to matrix column indices.
        disease_to_idx (dict): Mapping of disease names to matrix row indices.

    Returns:
        tuple: A tuple containing:
            - auc (float): The ROC Area Under the Curve score.
            - auprc (float): The Average Precision score.
            Returns (None, None) if no valid test pairs are found.
    """
    print("\n--- Evaluating Predictions ---")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get scores for all test pairs
    pred_scores = []
    true_labels = []
    
    for _, row in test_df.iterrows():
        try:
            dis_idx = disease_to_idx[row['DiseaseName']]
            drg_idx = drug_to_idx[row['ChemicalName']]
            
            score = Scores_matrix[dis_idx, drg_idx]
            pred_scores.append(score)
            true_labels.append(row['Label'])
        except KeyError:
            continue
    
    if len(pred_scores) == 0:
        print("Warning: no valid test pairs found.")
        return None, None
       
    # Calculate Metrics
    auc = roc_auc_score(true_labels, pred_scores)
    auprc = average_precision_score(true_labels, pred_scores)
    
    print(f"\n--- Validation Results ---")
    print(f"AUROC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # SAVE THE RAW DATA
    np.savez(DATA_FILE, scores=np.array(pred_scores), labels=np.array(true_labels))
    print(f"Raw evaluation data saved to '{DATA_FILE}'")
    
    return auc, auprc

def plot_score_distribution():
    """
    Generates and saves a histogram comparing the distribution of positive vs. negative scores.

    This function loads raw prediction data (scores and labels) from the `.npz` file 
    created by `evaluate_predictions`. It plots two overlapping histograms on a log 
    scale: one for known positive associations (Real Links) and one for negative 
    samples (Fake Links). This visualizes how well the model separates the two classes.

    The resulting plot is saved as a high-resolution PNG image.

    Raises:
        FileNotFoundError: If the input `.npz` data file does not exist (Phase 4 
            must be run first).
    """
    print(f"\nGenerating Score Distribution Plot...")
    
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. You must run Phase 4 once first.")
        return

    # Load Data
    data = np.load(DATA_FILE)
    pred_scores = data['scores']
    true_labels = data['labels']
    
    # Separate positives and negatives
    pos_scores = pred_scores[true_labels == 1]
    neg_scores = pred_scores[true_labels == 0]

    # Plot Setup (Clean Style - No Grid)
    sns.set_context("talk", font_scale=1.1)
    sns.set_style("ticks")
    
    plt.figure(figsize=(10, 6))
    
    # Histograms
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Negative (Fake Links)', 
             density=True, log=True, color='#4c72b0', edgecolor='none')
    plt.hist(pos_scores, bins=50, alpha=0.5, label='Positive (Real Links)', 
             density=True, log=True, color='#dd8452', edgecolor='none')
    
    plt.grid(False)
    plt.grid(visible=False, which='both', axis='both')
    
    plt.title('Prediction Score Distribution', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Diffusion Score', fontsize=16, labelpad=15)
    plt.ylabel('Density (Log Scale)', fontsize=16, labelpad=15)
    plt.legend(fontsize=14, loc='upper right', frameon=False)
    
    sns.despine()
    plt.tight_layout()
    
    # Save
    output_path = PLOT_DIR / "diffusion_score_distribution.png"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to: {output_path}")

# Allow running the plot function directly
if __name__ == "__main__":
    plot_score_distribution()