from setuptools import setup, find_packages

setup(
    name="whisper",
    version="0.1.0",
    description="Weak Heuristic Inference for Supervisory Protein intERaction mapping for PDB and AP-MS datasets",
    author="Vesal kasmaeifar",
    author_email="vesal.kasmaeifar@mail.utoronto.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    python_requires='>=3.9',
)
