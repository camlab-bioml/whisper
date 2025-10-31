# -- Project information -----------------------------------------------------
project = "whisper"
author = "Vesal Kasmaeifar"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # allow Markdown support
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# MyST (Markdown) options
myst_enable_extensions = ["linkify"]
