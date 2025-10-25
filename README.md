# whisper

`whisper` is a Python package for scoring protein–protein interactions from proximity labeling and affinity purification mass spectrometry datasets. It uses interpretable features, programmatic weak supervision, and decoy-based false discovery rate (FDR) estimation to identify high-confidence interactors.

## Installation

```bash
git clone https://github.com/camlab-bioml/whisper
cd whisper
pip install .
```

## Input Format

- A CSV file with:
  - One column named `Protein`
  - Other columns representing bait replicate intensities, named as `BAIT_1`, `BAIT_2`, etc.
- Control samples must be identifiable via substrings in their column names (e.g., `"EGFP"` or `"Empty"`).

## Usage

```python
#protein-level
from whisper.protein_features import feature_engineering_protein
from whisper.protein_train import train_and_score_protein
import pandas as pd


# Load intensity table
intensity_df = pd.read_csv("input_intensity_dataset.tsv", sep="\t")

controls = ['EGFP', 'Empty', 'NminiTurbo']

# Run feature engineering
features_df = feature_engineering_protein(intensity_df, controls)

# features_df = pd.read_csv("features.csv")


# Run scoring and FDR estimation
scored_df = train_and_score_protein(features_df, initial_positives=15, initial_negatives=200)


#peptide-level
from whisper.peptide_features import feature_engineering_peptide
from whisper.peptide_train import train_and_score_peptide
import pandas as pd


# Load intensity table
intensity_df = pd.read_csv("input_intensity_dataset.tsv", sep="\t")

controls = ['EGFP', 'Empty', 'NminiTurbo']

# Run feature engineering
features_df = feature_engineering_peptide(intensity_df, controls)

# features_df = pd.read_csv("features.csv")


# Run scoring and FDR estimation
scored_df = train_and_score_peptide(features_df, initial_positives=15, initial_negatives=200)


#fragment-level
from whisper.fragment_features import feature_engineering_fragment
from whisper.fragment_train import train_and_score_fragment
import pandas as pd


# Load intensity table
intensity_df = pd.read_csv("input_intensity_dataset.tsv", sep="\t")

controls = ['EGFP', 'Empty', 'NminiTurbo']

# Run feature engineering
features_df = feature_engineering_fragment(intensity_df, controls)

# features_df = pd.read_csv("features.csv")


# Run scoring and FDR estimation
scored_df = train_and_score_fragment(features_df, initial_positives=15, initial_negatives=200)
```

## Output

The final output includes:
- `predicted_probability`: Probability of each bait–prey interaction being real
- `FDR`: Estimated false discovery rate
- `global_cv_flag`: Flag for likely background preys based on variability across all samples

## Citation

This software is authored by: Vesal Kasmaeifar, Kieran R Campbell

Lunenfeld-Tanenbaum Research Institute & University of Toronto