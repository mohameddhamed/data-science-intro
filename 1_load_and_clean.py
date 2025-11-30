from enum import unique
from operator import le
from matplotlib import legend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import isfile, join
import seaborn as sns
import os

"""
FOLDER_PATH = "."

csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]
print("found these files: ", csv_files)

dfs = []
for file in csv_files:
    file_path = os.path.join(FOLDER_PATH, file)
    df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    dfs.append(df)

# concat all files into 1 dataframe
data = pd.concat(dfs, ignore_index=True)
print("Concatenation done")

data.columns = data.columns.str.strip()
print("fixed column names, 'Label' is now accessible")
"""
data: pd.DataFrame = pd.read_pickle("full_dataset_with_families.pkl")
print("Loaded pickle successfully")

print("Shape: ", data.shape)
print("\nColumns: ", list(data.columns))
print("\nFirst 5 rows: ")
print(data.head())

print("\nRaw label counts (top 20):")
print(data["Label"].value_counts().head(20))

label_mapping = {
    "BENIGN": "BENIGN",
    # DoS family
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    # DDoS
    "DDoS": "DDoS",
    # PortScan
    "PortScan": "PortScan",
    # Web attacks
    "Web Attack � Brute Force": "WebAttack",
    "Web Attack � XSS": "WebAttack",
    "Web Attack � Sql Injection": "WebAttack",
    # Brute force
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    # Botnet / Infiltration
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
}

# apply mapping
data["Label_family"] = data["Label"].map(label_mapping)

# check if anything is missing
print("\nMissing mappings: ", data["Label_family"].isna().sum())
if data["Label_family"].isna().any():
    print("Unmapped labels: ", data[data["Label_family"].isna()]["Label"].unique())


print("\nFinal attack family counts:")
family_counts = data["Label_family"].value_counts()
print(family_counts)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x=family_counts.index,
    y=family_counts.values,
    hue=family_counts.index,
    palette="viridis",
    legend=False,
)
plt.title("Distribution of Attack Families")
plt.ylabel("Number of flows")
plt.xticks(rotation=45)

for patch in ax.patches:
    patch.set_edgecolor("black")
    patch.set_linewidth(0.8)

for i, v in enumerate(family_counts.values):
    ax.text(i, v + 5000, f"{v:,}", ha="center", va="bottom", fontsize=10)


plt.tight_layout()
plt.savefig("label_distribution.png", dpi=300, bbox_inches="tight")
# plt.show()

# data.to_pickle("full_dataset_with_families.pkl")
