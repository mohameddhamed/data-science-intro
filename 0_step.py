# Install required packages
# !pip install scikit-learn pandas numpy matplotlib seaborn mlxtend

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import os
from google.colab import files
import warnings

warnings.filterwarnings("ignore")

# Set display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

print("✓ Setup complete! Now upload your MachineLearningCSV.zip file:")
print("Click the 'Choose Files' button below")

# Upload the dataset
uploaded = files.upload()

# Extract the ZIP file
for filename in uploaded.keys():
    if filename.endswith(".zip"):
        with ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall("dataset")
        print(f"✓ Extracted {filename}")

# List extracted files
print("\nExtracted files:")
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(".csv"):
            print(f"  - {file}")
