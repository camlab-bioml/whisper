# puppi

`puppi` is a Python package for scoring protein–protein interactions from proximity labeling and affinity purification mass spectrometry datasets. It uses interpretable features, positive-unlabeled (PU) learning, and decoy-based false discovery rate (FDR) estimation to identify high-confidence interactors.

## Installation

```bash
git clone https://github.com/camlab-bioml/puppi
cd puppi
pip install .
```

## Input Format

- A CSV file with:
  - One column named `Protein`
  - Other columns representing bait replicate intensities, named as `BAIT_1`, `BAIT_2`, etc.
- Control samples must be identifiable via substrings in their column names (e.g., `"EGFP"` or `"Empty"`).

## Usage

```python
from puppi.feature_engineering import run_feature_engineering
from puppi.train_and_fdr import run_training_and_fdr

import pandas as pd

# Step 1: Load data and run feature engineering
df = pd.read_csv("your_input_file.csv")
features_df = run_feature_engineering(df, control_keywords=["EGFP", "Empty"])

# Step 2: Train model and estimate FDR
results_df = run_training_and_fdr(features_df, initial_positives=10, initial_negatives=200)

# Save output
results_df.to_csv("puppi_results.csv", index=False)
```

## Output

The final output includes:
- `predicted_probability`: Probability of each bait–prey interaction being real
- `FDR`: Estimated false discovery rate
- `global_cv_flag`: Flag for likely background preys based on variability across all samples

## Citation

This software is authored by: Vesal Kasmaeifar, Kieran R Campbell

Lunenfeld-Tanenbaum Research Institute & University of Toronto