import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'results' / 'networks'
TABLES_DIR = BASE_DIR / 'results' / 'tables'
PLOT_DIR = BASE_DIR / 'results' / 'plots'
DATA_FILE = RESULTS_DIR / 'diffusion_results.npz'
REPORT_FILE = TABLES_DIR / 'model_performance.txt'

def evaluate_predictions(Scores_matrix, test_df, drug_to_idx, disease_to_idx, train_df=None):
    """
    Evaluates diffusion prediction quality on test data, training data (overfitting check),
    and a shuffled-score baseline (random predictor comparison).

    This function assesses link-prediction performance by:
    1. Scoring each (DiseaseName, ChemicalName) pair in `test_df` using the provided
       score matrix and index maps, collecting:
       - test_scores: predicted diffusion scores
       - test_labels: ground-truth labels (0/1) already present in `test_df`
    2. Computing TEST-set metrics:
       - ROC-AUC (discrimination)
       - AUPRC / Average Precision (precision–recall behavior, robust to class imbalance)
    3. Computing a RANDOM baseline by shuffling the test scores (labels fixed) and
       recalculating ROC-AUC, representing chance-level ordering performance.
    4. Optionally computing a TRAIN-set ROC-AUC as an overfitting check by:
       - extracting scores for training positives, then
       - sampling an equal-sized set of negative scores via `generate_negative_scores`,
         and evaluating ROC-AUC on the combined set.
    5. Writing a human-readable validation report to disk (including quick interpretation)
       and saving raw arrays (scores/labels) for downstream plotting.

    Args:
        Scores_matrix: Score lookup structure indexed as [disease_idx, drug_idx].
            Can be a NumPy array, SciPy sparse matrix, or any object supporting
            __getitem__ with two indices.
        test_df (pandas.DataFrame): Test samples produced by the splitter, expected to
            contain columns: 'DiseaseName', 'ChemicalName', and 'Label' (0/1).
        drug_to_idx (dict): Mapping from drug name -> column index into Scores_matrix.
        disease_to_idx (dict): Mapping from disease name -> row index into Scores_matrix.
        train_df (pandas.DataFrame, optional): Training positive edges with columns
            'DiseaseName' and 'ChemicalName'. If provided, a train ROC-AUC is computed
            for overfitting diagnostics. Default is None.

    Side Effects:
        - Ensures RESULTS_DIR, TABLES_DIR, and PLOT_DIR exist (mkdir).
        - Writes a text report to REPORT_FILE.
        - Saves raw test arrays to a .npz file via np.savez(DATA_FILE, ...).

    Returns:
        tuple:
            - test_auc (float): ROC-AUC on the labeled test set.
            - test_auprc (float): Average Precision (AUPRC) on the labeled test set.
        Returns (None, None) if no valid test pairs can be scored (e.g., mapping mismatches).
    """

    print("\n--- Evaluating Predictions ---")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate TEST Set ---
    # Since this used data_splitter module, test_df ALREADY has 'Label' (0 and 1)
    test_scores = []
    test_labels = []
    
    for _, row in test_df.iterrows():
        try:
            dis_idx = disease_to_idx[row['DiseaseName']]
            drg_idx = drug_to_idx[row['ChemicalName']]
            
            score = float(Scores_matrix[dis_idx, drg_idx])
            test_scores.append(score)
            test_labels.append(row['Label'])
        except KeyError:
            continue
    
    if not test_scores:
        print("Warning: No valid test pairs found.")
        return None, None
        
    test_auc = roc_auc_score(test_labels, test_scores)
    test_auprc = average_precision_score(test_labels, test_scores)

    # Random Baseline 
    print("Calculating Random Baseline (Shuffled Predictions)...")
    shuffled_scores = test_scores.copy()
    random.shuffle(shuffled_scores) # Randomize the order
    random_auc = roc_auc_score(test_labels, shuffled_scores)
    
    # Evaluate TRAIN Set (for Overfitting Check)
    train_auc = None
    if train_df is not None:
        print("Calculating AUC for TRAIN set (Checking for Overfitting)...")
        train_pos_scores = []
        for _, row in train_df.iterrows():
            try:
                dis_idx = disease_to_idx[row['DiseaseName']]
                drg_idx = drug_to_idx[row['ChemicalName']]
                train_pos_scores.append(float(Scores_matrix[dis_idx, drg_idx]))
            except KeyError:
                continue
        
        # Generate Negatives for Training
        if train_pos_scores:
            # Exclude BOTH Train and Test positives so negatives are truly unknown
            all_known = pd.concat([train_df, test_df], ignore_index=True)
            
            train_neg_scores = generate_negative_scores(
                Scores_matrix, len(train_pos_scores), 
                exclude_df=all_known,  
                drug_to_idx=drug_to_idx, 
                disease_to_idx=disease_to_idx
            )
            
            y_train = [1] * len(train_pos_scores) + [0] * len(train_neg_scores)
            y_scores_train = train_pos_scores + train_neg_scores
            
            train_auc = roc_auc_score(y_train, y_scores_train)

    # Report & Save
    print(f"\n--- Final Validation Results ---")
    report_lines = ["--- Diffusion Model Performance ---"]
    
    # Print to Terminal
    if train_auc is not None:
        print(f"Train AUC:  {train_auc:.4f} (Checking Overfitting)")
    print(f"Test AUC:   {test_auc:.4f}  (Real Performance)")
    print(f"Random AUC: {random_auc:.4f} (Baseline / Chance)")
    print(f"Test AUPRC: {test_auprc:.4f}")
    
    # Save to File
    if train_auc is not None:
        report_lines.append(f"Train AUC: {train_auc:.4f}")
    report_lines.append(f"Test AUC:  {test_auc:.4f}")
    report_lines.append(f"Random Baseline AUC: {random_auc:.4f}")
    report_lines.append(f"Test AUPRC: {test_auprc:.4f}")
    
    report_lines.append("-" * 30)
    report_lines.append("INTERPRETATION:")
    
    # Overfitting
    if train_auc is not None:
        diff = train_auc - test_auc
        if diff > 0.1:
            report_lines.append(f"[WARNING] Overfitting: Train is much better than Test (Diff: {diff:.4f})")
        else:
            report_lines.append(f"[OK] Generalization: Train and Test are similar (Diff: {diff:.4f})")

    # Better than Random?
    if test_auc > random_auc + 0.2:
         report_lines.append(f"[OK] Performance: Model significantly beats random guessing ({test_auc:.2f} vs {random_auc:.2f})")
    else:
         report_lines.append("[FAIL] Performance: Model is not much better than random guessing.")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Full report saved to '{REPORT_FILE}'")
    
    # Save raw data for plotting
    np.savez(DATA_FILE, scores=np.array(test_scores), labels=np.array(test_labels))
    
    return test_auc, test_auprc

def generate_negative_scores(Scores_matrix, n_samples, exclude_df, drug_to_idx, disease_to_idx):
    """
    Samples diffusion scores for randomly-chosen negative (unobserved) disease–drug pairs.

    This helper function is used to construct a negative class for evaluation by:
    1. Converting the observed links in `exclude_df` into an index-based set of known pairs
       (disease_idx, drug_idx), using the provided mapping dictionaries.
    2. Randomly sampling (row, col) indices from the full score matrix.
    3. Keeping only sampled pairs that are NOT in the known pair set (i.e., treating them
       as negatives/unobserved links).
    4. Collecting and returning the corresponding diffusion scores until `n_samples` are
       obtained or a maximum number of attempts is reached (to avoid infinite loops on
       dense matrices).

    Args:
        Scores_matrix: Score matrix indexed as [disease_idx, drug_idx]. Must expose `.shape`
            and support two-index access (e.g., NumPy array or SciPy sparse matrix).
        n_samples (int): Number of negative scores to sample.
        exclude_df (pandas.DataFrame): Edge list of known/observed pairs to exclude from
            negative sampling. Must include 'DiseaseName' and 'ChemicalName' columns.
        drug_to_idx (dict): Mapping from drug name -> column index.
        disease_to_idx (dict): Mapping from disease name -> row index.

    Notes:
        - Sampling is uniform over all matrix coordinates, not degree-matched.
        - The function caps sampling attempts at `n_samples * 20` to guarantee termination.
          If the matrix is very dense or exclude_df is very large, fewer than `n_samples`
          negatives may be returned.

    Returns:
        list: A list of sampled negative scores (length <= n_samples).
    """

    known_pairs = set()
    for _, row in exclude_df.iterrows():
        try:
            u = disease_to_idx[row['DiseaseName']]
            v = drug_to_idx[row['ChemicalName']]
            known_pairs.add((u, v))
        except KeyError:
            continue
            
    neg_scores = []
    max_dis = Scores_matrix.shape[0] - 1
    max_drg = Scores_matrix.shape[1] - 1
    
    count = 0
    max_attempts = n_samples * 20
    attempts = 0
    
    while count < n_samples and attempts < max_attempts:
        attempts += 1
        rd = random.randint(0, max_dis)
        rc = random.randint(0, max_drg)
        
        if (rd, rc) not in known_pairs:
            neg_scores.append(float(Scores_matrix[rd, rc]))
            count += 1
            
    return neg_scores

def plot_score_distribution():
    """
    Plots the distribution of diffusion prediction scores for positives vs negatives (test set).

    This function creates a diagnostic histogram to visually assess how well the diffusion
    model separates real (positive) associations from fake (negative) ones by:
    1. Loading saved evaluation arrays (scores and labels) from DATA_FILE (NPZ format).
    2. Splitting predicted scores into two groups using the ground-truth labels:
       - Positive (Label = 1): real/held-out links
       - Negative (Label = 0): sampled non-links
    3. Plotting both score distributions on the same histogram using:
       - density normalization (comparability between groups)
       - log-scaled y-axis (to show tail behavior)
    4. Saving the figure to the plots directory.

    Notes:
        - Expects DATA_FILE to be created by `evaluate_predictions()` via `np.savez(...)`
          with arrays named 'scores' and 'labels'.
        - Uses Seaborn styling for presentation-quality plots.
        - The histogram uses log=True on the y-axis; scores can be highly skewed.

    Returns:
        None: Saves the figure to PLOT_DIR / "diffusion_score_distribution.png".
    """
    
    print(f"\nGenerating Score Distribution Plot...")
    
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found.")
        return

    data = np.load(DATA_FILE)
    pred_scores = data['scores']
    true_labels = data['labels']
    
    pos_scores = pred_scores[true_labels == 1]
    neg_scores = pred_scores[true_labels == 0]

    sns.set_context("talk", font_scale=1.1)
    sns.set_style("ticks")
    
    plt.figure(figsize=(10, 6))
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Negative (Fake)', 
             density=True, log=True, color='#4c72b0', edgecolor='none')
    plt.hist(pos_scores, bins=50, alpha=0.5, label='Positive (Real)', 
             density=True, log=True, color='#dd8452', edgecolor='none')
    
    plt.title('Prediction Score Distribution', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Diffusion Score', fontsize=16)
    plt.ylabel('Density (Log Scale)', fontsize=16)
    plt.legend(fontsize=14, frameon=False)
    
    sns.despine()
    plt.tight_layout()
    
    output_path = PLOT_DIR / "diffusion_score_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_score_distribution()