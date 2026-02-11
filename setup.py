import os

# The industry-standard folder structure
structure = {
    "folders": [
        "data/raw",          # Original, immutable data
        "data/processed",    # Cleaned data for modeling
        "notebooks",         # Jupyter notebooks
        "src",               # specific scripts (cleaning, plotting)
        "models",            # Saved .pkl models
        "app",               # Streamlit dashboard code
    ],
    "files": [
        "README.md",         # Project documentation
        ".gitignore",        # Files to exclude from Git
        "requirements.txt",  # Library dependencies
        "src/__init__.py",   # Makes src a Python package
        "app/main.py"        # Dashboard entry point
    ]
}

# Create Folders
for folder in structure["folders"]:
    os.makedirs(folder, exist_ok=True)
    print(f"[OK] Created folder: {folder}")

# Create Files
for file in structure["files"]:
    with open(file, 'w') as f:
        pass  # Create empty file
    print(f"[OK] Created file: {file}")

print("\nSUCCESS: Project structure created! You can now delete this script.")