# whisper

[![PyPI](https://img.shields.io/pypi/v/whisper-ppi.svg?color=brightgreen)](https://pypi.org/project/whisper-ppi/)
[![Docs](https://readthedocs.org/projects/whisper/badge/?version=latest)](https://whisper.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/pypi/pyversions/whisper-ppi.svg)](https://pypi.org/project/whisper-ppi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![logo](https://github.com/camlab-bioml/whisper/blob/main/logo.png)

`whisper-ppi` is a Python package for scoring protein–protein interactions from proximity labeling and affinity purification mass spectrometry datasets.  
It uses interpretable features, programmatic weak supervision, and decoy-based false discovery rate (FDR) estimation to identify high-confidence interactors.

More details of the algorithm and benchmarking can be found in the manuscript _Predicting protein interaction and proximity evidence with weakly supervised learning (Kasmaeifar et al., 2026)_.

---

## Overview
![WHISPER Overview](https://github.com/camlab-bioml/whisper/blob/main/overview_figure.png)

---

## System Requirements

### Software dependencies

| Package | Minimum version |
|---------|----------------|
| Python | 3.10 |
| numpy | 1.24 |
| pandas | 1.5 |
| scikit-learn | 1.2 |
| scipy | 1.10 |

### Tested operating systems

- macOS 13 and later
- Windows 10 and later

### Tested Python versions

Python 3.10, 3.11, and 3.12.

### Non-standard hardware

No non-standard hardware is required. All analyses can be run on a standard desktop or laptop computer.

---

## Installation

### Install from PyPI

```bash
pip install whisper-ppi
```

### Install from GitHub

```bash
git clone https://github.com/camlab-bioml/whisper
cd whisper
pip install .
```

**Typical install time:** Installation from PyPI takes approximately 1–2 minutes on a standard desktop computer with a normal internet connection.

---

## Input Format

- A CSV file with:
  - One column named `Protein`
  - Other columns representing bait replicate intensities, named as `BAIT_1`, `BAIT_2`, etc.
- Control samples must be identifiable via substrings in their column names (e.g., `"EGFP"` or `"Empty"`).

---

## Demo

A self-contained tutorial notebook is provided in the [`tutorial/`](tutorial/) directory. It uses the bundled example dataset (`input_intensity_dataset.tsv`) and walks through feature engineering and scoring at the protein level.

### Running the demo

```bash
cd tutorial
jupyter notebook tutorial_whisper.ipynb
```

### Expected output

Running the notebook produces:

- `features.csv` — engineered features for each bait–prey pair
- `whisper_protein_scores.csv` — scored interactions with `predicted_probability` and `FDR` columns

Example rows from `whisper_protein_scores.csv`:

| Protein | predicted_probability | FDR | global_cv_flag |
|---------|-----------------------|-----|----------------|
| PROT_A  | 0.94                  | 0.01 | False         |
| PROT_B  | 0.21                  | 0.48 | True          |

### Expected run time

The full tutorial notebook runs in approximately **[5-10 minutes]** on a standard desktop computer (no GPU required).

---

## Instructions for Use

### Running on your own data

```python
# Protein-level
from whisper.protein_features import feature_engineering_protein
from whisper.protein_train import train_and_score_protein
import pandas as pd

# Load intensity table
intensity_df = pd.read_csv("input_intensity_dataset.tsv", sep="\t")

controls = ['EGFP', 'Empty', 'NminiTurbo']

# Run feature engineering
features_df = feature_engineering_protein(intensity_df, controls)

# Optionally save features to reuse without recomputing
features_df.to_csv("features.csv", index=False)
features_df = pd.read_csv("features.csv")

# Run scoring and FDR estimation
scored_df = train_and_score_protein(features_df, initial_positives=15, initial_negatives=200)
```

```python
# Peptide-level
from whisper.peptide_features import feature_engineering_peptide
from whisper.peptide_train import train_and_score_peptide
import pandas as pd

intensity_df = pd.read_csv("input_intensity_dataset.tsv", sep="\t")
controls = ['EGFP', 'Empty', 'NminiTurbo']

features_df = feature_engineering_peptide(intensity_df, controls)
scored_df = train_and_score_peptide(features_df, initial_positives=15, initial_negatives=200)
```

```python
# Fragment-level
from whisper.fragment_features import feature_engineering_fragment
from whisper.fragment_train import train_and_score_fragment
import pandas as pd

intensity_df = pd.read_csv("input_intensity_dataset.tsv", sep="\t")
controls = ['EGFP', 'Empty', 'NminiTurbo']

features_df = feature_engineering_fragment(intensity_df, controls)
scored_df = train_and_score_fragment(features_df, initial_positives=15, initial_negatives=200)
```

### Reproducing manuscript results

To reproduce all quantitative results from the manuscript, apply `whisper-ppi` to the datasets described in the Methods section using the parameters reported therein. The tutorial notebook demonstrates the standard workflow on a representative example dataset.

Full documentation is available at [whisper.readthedocs.io](https://whisper.readthedocs.io/en/latest/).

---

## Output

The final output includes:

| Column | Description |
|--------|-------------|
| `predicted_probability` | Probability of the bait–prey interaction being real |
| `FDR` | Estimated false discovery rate |
| `global_cv_flag` | Flag for likely background preys based on variability across all samples |
| Feature columns | Individual feature values used in scoring |

---

## Citation

If you use `whisper-ppi` in your work, please cite:

> Kasmaeifar V, Campbell KR. _Predicting protein interaction and proximity evidence with weakly supervised learning._ 2026.

This software is authored by: Vesal Kasmaeifar, Kieran R Campbell  
Lunenfeld-Tanenbaum Research Institute & University of Toronto

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
