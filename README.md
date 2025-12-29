# Computational Analysis of Drug-Disease Associations

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python 3.10+"></a>
  <a href="https://github.com/MPSVIRAJ/Disease-Drug-Networks"><img src="https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge" alt="Maintained"></a>
  <a href="https://github.com/MPSVIRAJ/Disease-Drug-Networks/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT"></a>
</p>

<p align="center">
  A hybrid network medicine framework integrating bipartite topology, centrality analysis, and network diffusion modeling to systematically identify novel therapeutic indications for existing drugs.
</p>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#the-approach">The Approach</a></li>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#pipeline-workflow">Pipeline Workflow</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li>
      <a href="#the-dataset">The Dataset</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#phase-1-network-construction">Phase 1: Network Construction</a></li>
        <li><a href="#phase-2-core-analysis">Phase 2: Core Analysis</a></li>
        <li><a href="#phase-3-visualization">Phase 3: Visualization</a></li>
        <li><a href="#phase-4-predictive-modeling">Phase 4: Predictive Modeling</a></li>
        <li><a href="#full-pipeline">Full Pipeline</a></li>
      </ul>
    </li>
    <li><a href="#results">Results</a></li>
    <li>
      <a href="#limitations">Limitations</a>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

## About The Project

The high cost and extended timelines of traditional drug discovery have shifted focus toward drug repurposing. This project establishes a computational framework based on **Network Medicine** to systematically identify novel therapeutic indications for existing drugs using the Comparative Toxicogenomics Database (CTD).

### The Approach
This framework utilizes a multi-stage graph analysis pipeline:
* **Bipartite Construction:** Models relationships as a Drug-Disease bipartite graph.
* **Topological Filtering:** Uses Betweenness Centrality to distinguish therapeutic "bridge" nodes from environmental toxicants (which often act as high-degree hubs).
* **Network Pruning:** Addresses the biological "hairball effect" by pruning low-weight edges in projected networks.
* **Diffusion Modeling:** Implements a "Guilt-by-Association" network diffusion algorithm ($R = B \cdot S$) to predict unobserved links.

### Key Features
* **End-to-End Pipeline:** From raw data parsing to final AUROC evaluation.
* **Dimensionality Reduction:** Uses `node2vec` and `UMAP` to visualize high-dimensional disease embeddings.
* **Robust Pruning:** Implements weight-based pruning (90th percentile) to handle dense biological networks.
* **Metric Validation:** Evaluates models using AUROC and AUPRC with a rigorous 90/10 hold-out split (preventing data leakage).
* **Reproducible Analysis:** Modular codebase allowing independent execution of analysis phases.

### Built With
This project was built using the following major libraries:

* [Python](https://www.python.org/) - Core logic
* [NetworkX](https://networkx.org/) - Graph construction and centrality algorithms
* [NodeVectors](https://github.com/VHRanger/nodevectors) - Fast implementation of Node2Vec for embeddings
* [UMAP](https://umap-learn.readthedocs.io/) - Uniform Manifold Approximation and Projection
* [Scikit-learn](https://scikit-learn.org/) - Metric evaluation (ROC/AUC) and normalization
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data manipulation
* [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization

## Pipeline Workflow

The project is structured into four distinct computational phases:

1.  **Network Construction:** Parsing CTD data, building the bipartite graph $G=(U,V,E)$, and projecting it into Drug-Drug and Disease-Disease unipartite networks.
2.  **Core Analysis:** Calculating centrality metrics (Betweenness, Eigenvector) and identifying disease communities using the Louvain algorithm.
3.  **Advanced Validation:** Generating node embeddings via random walks (Node2Vec) and visualizing clusters using UMAP.
4.  **Diffusion Method:** Running the prediction algorithm, performing the train-test split, and generating the final ranked list of repurposing candidates.

## Repository Structure

```text
Disease-Drug-Networks/
├── data/                   # Place CTD dataset here
├── results/
│   ├── networks/           # Stores .pkl graph files (generated)
│   ├── plots/              # Stores generated figures (UMAP, Histograms)
│   └── tables/             # Stores CSV results (Predictions, Stats)
├── src/                    # Source code modules
│   ├── data_loader.py      # Data parsing
│   ├── network_builder.py  # Graph projection & pruning
│   ├── diffusion_model.py  # Prediction logic
│   └── ...
├── run_analysis.py         # Main execution script
├── get_final_results.py    # Utility to extract tables
├── update_community_plot.py# Utility to refine plots
└── requirements.txt        # Dependencies
```
### The Dataset
This study utilizes the Comparative Toxicogenomics Database (CTD). Due to file size limitations and licensing, the raw data is not included in this repository.

Instructions:

1. Go to CTD Downloads.
2. Download the "Chemical-disease associations" file (CTD_chemicals_diseases.tsv.gz).
3. Extract the file.
4. Rename it to CTD_chemicals_diseases.csv and place it in the data/ directory.

## Getting Started
### Prerequisites
* Python 3.10+
* Git

### Installation
**Clone the repository:**

```sh

git clone [https://github.com/MPSVIRAJ/Disease-Drug-Networks.git](https://github.com/MPSVIRAJ/Disease-Drug-Networks.git)
cd Disease-Drug-Networks
```
**Create and activate a virtual environment (Optional):**
    
    **On macOS/Linux:**
    ```sh
    python3 -m venv ddn
    source ddn/bin/activate 
    ```

    **On Windows:**
    
    Make the environment 
    ```sh
    python -m venv ddn     
    ```
    Before activating, you may need to run this command in PowerShell:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    ```
    Activate the environment:
    ```powershell
    .\ddn\Scripts\activate
    ```
**Install dependencies:**

```sh
pip install -r requirements.txt
```

## Usage
The pipeline is controlled via the run_analysis.py script. You can run individual phases or the entire workflow.

* Phase 1: Network Construction
Loads data and builds the bipartite and projected graphs.

```sh
python run_analysis.py --phase1
```
* Phase 2: Core Analysis
Calculates centrality measures (to identify bridge drugs) and detects disease communities.

```sh
python run_analysis.py --phase2
```

* Phase 3: Visualization
Generates Node2Vec embeddings and creates UMAP plots to visualize disease clustering.
```sh
python run_analysis.py --phase3
```

* Phase 4: Predictive Modeling
Runs the network diffusion algorithm, evaluates AUROC/AUPRC, and outputs prediction scores.
```sh
python run_analysis.py --phase4
```

* Full Pipeline
To run the complete analysis from start to finish:
```sh
python run_analysis.py --all
```
**Note:** After running the pipeline, use the helper script to generate the final CSV tables:
```sh
python get_final_results.py
```
## Results
Key findings generated by this pipeline include:

Topological Density: The projected disease-disease network exhibited a "hairball" effect (Density ≈ 0.77), which was successfully mitigated via weight-based pruning.

Toxicant Filtering: Betweenness Centrality analysis successfully distinguished therapeutic agents (e.g., Methotrexate, Amphotericin B) from environmental toxicants (e.g., Bisphenol A), which dominated simple degree-based rankings.

Prediction Accuracy: The diffusion model achieved an AUROC of 0.93 on the hold-out test set.

**Novel Candidates:**

Valproic Acid was identified as a strong candidate for Malaria (supported by HDAC inhibition mechanisms).

A cluster of neurological agents (e.g., Phenytoin) was predicted for Opiate Overdose.

Visual outputs (UMAP clusters, score distributions) can be found in the results/plots/ directory.

## Limitations
Toxicant Bias: The CTD contains environmental associations. While centrality filtering helps, some high-degree toxicants may still appear in prediction lists due to the "hub" effect.

Mechanism Agnostic: The diffusion model relies on "Guilt-by-Association." It predicts that a link exists based on topology but does not explain the molecular mechanism (e.g., gene expression vs. protein binding).

Computational Intensity: Generating embeddings for the full unpruned network is computationally expensive. The pipeline uses a pruning threshold (90th percentile) to manage this, which may discard subtle but valid signals.

## License
Distributed under the MIT License. See LICENSE for more information.

## Citation
If you use this code or methodology, please cite:

Malwaththa Pathirannehelage, S. V. (2025). Computational Analysis of Drug-Disease Associations: A Hybrid Network Approach for Drug Repurposing.

<p align="center"> <br /> <a href="https://www.google.com/search?q=https://github.com/MPSVIRAJ/Disease-Drug-Networks/issues">Report Bug</a> </p>