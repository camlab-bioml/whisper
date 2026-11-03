from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("whisper-ppi")
except PackageNotFoundError:   # running from a source checkout that is not installed
    __version__ = "unknown"

from .protein_features import feature_engineering_protein as feature_engineering_protein
from .protein_train import train_and_score_protein as train_and_score_protein

from .peptide_features import feature_engineering_peptide
from .peptide_train import train_and_score_peptide

from .fragment_features import feature_engineering_fragment
from .fragment_train import train_and_score_fragment

__all__ = [
    "__version__",
    "feature_engineering_protein", "train_and_score_protein",
    "feature_engineering_peptide", "train_and_score_peptide",
    "feature_engineering_fragment", "train_and_score_fragment",
]
