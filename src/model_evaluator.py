# src/model_evaluator.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_predictions(Scores_matrix, test_df, drug_to_idx, disease_to_idx, plot_output_dir):
    """
    Validates the diffusion scores against the test set.
    """
    print("\n--- Evaluating Predictions ---")
    
    # 1. Get scores for all test pairs
    pred_scores = []
    true_labels = []
    
    for _, row in test_df.iterrows():
        try:
            # Get the matrix indices for the disease and drug
            dis_idx = disease_to_idx[row['DiseaseName']]
            drg_idx = drug_to_idx[row['ChemicalName']]
            
            # Get the predicted score from the matrix
            score = Scores_matrix[dis_idx, drg_idx]
            
            pred_scores.append(score)
            true_labels.append(row['Label'])
            
        except KeyError:
            # This can happen if a drug/disease was in test set but not train set
            # (due to our simple split, but it's rare and okay to skip)
            continue
    
    # Check if we have any valid scores
    if len(pred_scores) == 0:
        print("Warning: no valid test pairs found in evaluation.")
        return None, None
       
    # 2. Calculate performance metrics
    auc = roc_auc_score(true_labels, pred_scores)
    auprc = average_precision_score(true_labels, pred_scores)
    
    print(f"\n--- Validation Results ---")
    print(f"Area Under ROC Curve (AUROC): {auc:.4f}")
    print(f"Area Under PR Curve (AUPRC): {auprc:.4f}")
    
    # 3. Plot score distributions
    pos_scores = [s for s, t in zip(pred_scores, true_labels) if t == 1]
    neg_scores = [s for s, t in zip(pred_scores, true_labels) if t == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Negative (Fake Links)', density=True, log=True)
    plt.hist(pos_scores, bins=50, alpha=0.5, label='Positive (Real Links)', density=True, log=True)
    plt.legend()
    plt.title('Prediction Score Distribution')
    plt.xlabel('Diffusion Score')
    plt.ylabel('Density (Log Scale)')
    plot_path = plot_output_dir / "diffusion_score_distribution.png"
    plt.savefig(plot_path)
    
    print(f"Score distribution plot saved to '{plot_path}'")
    return auc, auprc