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

"""Descriptive Statistics"""
data: pd.DataFrame = pd.read_pickle("full_dataset_with_families.pkl")
# target
y = data["Label_family"]
# features
x = data.drop(["Label", "Label_family"], axis=1)

print("\nMissing values:")
print(data.isnull().sum().sum())
print("\nInfinity values:")
inf_cols = x.columns[(x == np.inf).any() | (x == -np.inf).any()]
print(f"Columsn with +- inf: {list(inf_cols)}")

# replacing infs with NaN
x.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Replaced inf -> NaN")

print("Filling NaN values with column median...")
for col in x.columns:
    if x[col].isnull().any():
        median_val = x[col].median()  # because median is robust to outliers
        x[col].fillna(median_val, inplace=True)

print("Final NaN count after cleaning: ", x.isnull().sum().sum())  # should be 0

print("\nSample descriptive statistics:")
print(x.describe().T[["mean", "std", "min", "50%", "max"]].head(10))

"""Outlier detection - boxplots"""

key_features = [
    "Flow Duration",
    "Total Length of Fwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
]

plt.figure(figsize=(15, 10))
for i, feat in enumerate(key_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(
        y=x[feat], x=y, hue=y, palette="Set2", legend=False
    )  # 1 boxplot per attack family
    plt.title(f"{feat} by Attack Family", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel(feat)
plt.tight_layout()
plt.savefig("key_feature_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()

"""Univariate analysis - histograms"""

plt.figure(figsize=(15, 10))
hist_features = [
    "Flow Duration",
    "Flow Bytes/s",
    "Init_Win_bytes_forward",
    "Destination Port",
]

for i, feat in enumerate(hist_features, 1):
    plt.subplot(2, 2, i)
    # log scale because values span many orders of magnitude
    x[feat].hist(bins=50, log=True, alpha=0.7, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {feat} (log scale)")
    plt.xlabel(feat)
    plt.ylabel("Count (log)")
plt.tight_layout()
plt.savefig("key_feature_histograms.png", dpi=300)
plt.show()

"""Correlation anaylysis"""
print("\n Finding highly correlated features...")
# full correlation matrix
corr = x.corr()
high_corr = corr.abs().unstack().sort_values(ascending=False)
high_corr = high_corr[high_corr > 0.9]  # keep only high correlations
high_corr = high_corr[high_corr < 1]  # remove self-correlation

print("Top 15 pairs with correlation > 0.9:")
print(high_corr.head(15))

top_corr_pairs = high_corr.head(40)
top_features = set()
for idx in top_corr_pairs.index:
    # collect all unique feature names
    top_features.update(idx)
top_features = list(top_features)[:20]  # limit to first 20 for heatmap

plt.figure(figsize=(12, 10))
sns.heatmap(
    x[top_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.8},
    square=True,
    linewidths=0.5,
    center=0,
)
plt.title("Correlation Heatmap of Top 20 Correlated Features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

"""Multivariate - pca 2d & 3d"""
print("\n PCA Visualization...")
# pca requires scaled data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(
    x
)  # scale all features to mean=0, std=1 (unit variance)

# 2-component PCA
pca2 = PCA(n_components=2)
x_pca2 = pca2.fit_transform(x_scaled)
print(f"PCA 2D explained variance: {pca2.explained_variance_ratio_.sum():.4f}")

# 3-component PCA
pca3 = PCA(n_components=3)
x_pca3 = pca3.fit_transform(x_scaled)
print(f"PCA 3D explained variance: {pca3.explained_variance_ratio_.sum():.4f}")

# 2D scatter plot
plt.figure(figsize=(12, 9))
label_codes = y.astype("category").cat.codes
unique_labels = y.astype("category").cat.categories

scatter = plt.scatter(
    x_pca2[:, 0],
    x_pca2[:, 1],
    c=label_codes,
    cmap="tab10",
    alpha=0.6,
    s=2,
    edgecolors="none",
)
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=plt.cm.tab10(i / len(unique_labels)), label=label)
    for i, label in enumerate(unique_labels)
]
plt.legend(
    handles=legend_elements,
    title="Attack Family",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.title("PCA 2D - Attack Families Separability")
plt.xlabel(f"PCA Component 1 ({pca2.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PCA Component 2 ({pca2.explained_variance_ratio_[1]:.1%} variance)")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=300, bbox_inches="tight")
plt.show()

# 3D scatter plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    x_pca3[:, 0],
    x_pca3[:, 1],
    x_pca3[:, 2],
    c=y.astype("category").cat.codes,
    cmap="tab10",
    alpha=0.6,
    s=2,
)
legend_elements = [
    Patch(facecolor=plt.cm.tab10(i / len(unique_labels)), label=label)
    for i, label in enumerate(unique_labels)
]
ax.legend(handles=legend_elements, title="Attack Family", loc="upper left")
ax.set_title("PCA 3D - Attack Families Separability")
ax.set_xlabel(f"PCA Component 1 ({pca3.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PCA Component 2 ({pca3.explained_variance_ratio_[1]:.1%} variance)")
ax.set_zlabel(f"PCA Component 3 ({pca3.explained_variance_ratio_[2]:.1%} variance)")
plt.tight_layout()
plt.savefig("pca_3d.png", dpi=300, bbox_inches="tight")
plt.show()
