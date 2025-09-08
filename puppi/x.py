from features import feature_engineering
from train import train_and_score
import pandas as pd


# Load intensity table
intensity_df = pd.read_csv("../tutorial/input_intensity_dataset.tsv", sep="\t")

controls = ['EGFP', 'Empty', 'NminiTurbo']

# Run feature engineering
features_df = feature_engineering(intensity_df, controls)

# features_df = pd.read_csv("features.csv")


# Run PU learning and FDR estimation
scored_df = train_and_score(features_df, initial_positives=15, initial_negatives=200)
