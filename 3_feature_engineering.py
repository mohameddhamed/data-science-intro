import time
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Goal: Clean data → remove junk → scale → reduce dimensions → create several high-quality feature sets
# Output: Multiple ready-to-use datasets + comparison table + plots for report

data: pd.DataFrame = pd.read_pickle("full_dataset_with_families.pkl")
print("Loaded pickle successfully")

# separate target (multi-class attack family) and features
y = data["Label_family"]  # what we want to predict
x_raw = data.drop(["Label", "Label_family"], axis=1)

""" Final cleaning: handle inf and NaN values """
x = x_raw.copy()
x.columns = x.columns.str.strip()  # remove any leading/trailing spaces in column names

print(f"Replacing inf values with NaN...")
x.replace(
    [np.inf, -np.inf], np.nan, inplace=True
)  # Flow Bytes often has inf when duration=0

print("Filling NaN values with column median...")
x.fillna(x.median(), inplace=True)  # median is robust to outliers

print(f"After final cleaning, any NaN left? {x.isnull().sum().sum()} (should be 0)")

""" Remove exact duplicate columns - OPTIMIZED VERSION """
# Instead of transposing the entire massive dataframe, we'll use a hash-based approach
print("Removing exact duplicate columns (optimized method)...")
start_time = time.time()

# First, remove columns with duplicate names
x = x.loc[:, ~x.columns.duplicated()]
print(f"Removed duplicate column names. Shape: {x.shape}")

# Then, find columns with identical content using hashing (much faster)
# Sample a subset of rows for initial duplicate detection (faster)
sample_size = min(10000, len(x))
x_sample = x.sample(n=sample_size, random_state=42)

# Create hash for each column based on sample
column_hashes = {}
duplicate_cols = []
cols_to_keep = []

for col in x.columns:
    # Create a hash of the column's values in the sample
    col_hash = hash(tuple(x_sample[col].values))

    if col_hash in column_hashes:
        # Potential duplicate - verify with full column comparison
        original_col = column_hashes[col_hash]
        if x[col].equals(x[original_col]):  # exact duplicate
            duplicate_cols.append(col)  # mark for removal
            print(f"  Found duplicate: '{col}' is identical to '{original_col}'")
        else:
            column_hashes[col_hash] = col  # different content, keep it
            cols_to_keep.append(col)  # keep it
    else:
        column_hashes[col_hash] = col  # new hash, keep it
        cols_to_keep.append(col)  # keep it

# Keep only non-duplicate columns
if duplicate_cols:
    x = x[cols_to_keep]  # retain only unique columns
    print(f"Removed {len(duplicate_cols)} duplicate columns: {duplicate_cols}")
else:
    print("No duplicate columns found")

print(f"Duplicate removal completed in {time.time() - start_time:.1f} seconds")

# Also manually remove known junk columns (all zeros or irrelevant)
manual_drops = (
    ["Fwd Header Length.1"] if "Fwd Header Length.1" in x.columns else []
)  # known junk
if manual_drops:  # drop if exists
    x.drop(columns=manual_drops, inplace=True)
    print(f"Manually dropped: {manual_drops}")

print(f"Final feature set shape after cleaning: {x.shape[1]} features")

""" Standard Scaling (required for SFS, PCA, distance-based models) """
print("\nApplying StandardScaler (mean=0, std=1) to features...")
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)  # returns a numpy array
x_scaled_df = pd.DataFrame(
    x_scaled, columns=x.columns
)  # convert back to DataFrame for SFS

# saving cleaned and scaled data for next steps to transform test data
joblib.dump(scaler, "standard_scaler.pkl")
print("Saved StandardScaler as 'standard_scaler.pkl'")

""" Sequential Feature Selection (SFS, SBS, Bidirectional) """
# Using Logistic Regression with 'saga' solver -> fast + supports multiclass natively
base_model_for_selection = LogisticRegression(
    multi_class="multinomial",  # Handles 8 classes directly
    solver="saga",  # fast for large datasets
    max_iter=1000,  # enough for convergence in selection
    n_jobs=-1,  # utilize all CPU cores
    random_state=42,  # for reproducibility
)

""" Sequential Forward Selection (SFS) (start from 0 -> add best features one by one) """
from sklearn.feature_selection import SequentialFeatureSelector as SFS

print("\nStarting Sequential Forward Selection (SFS) -> selecting 20 best features...")
start = time.time()
sfs = SFS(
    estimator=base_model_for_selection,  # using logistic regression as base model
    n_features_to_select=20,  # select top 20 features
    direction="forward",  # SFS
    scoring="accuracy",  # optimize for accuracy
    cv=3,  # 3-fold cross-validation (faster than 5-fold)
    n_jobs=-1,  # utilize all CPU cores
)
sfs.fit(x_scaled_df, y)  # fit SFS on scaled data
sfs_features = x.columns[sfs.get_support()].tolist()  # get selected feature names
print(f"SFS finished in {(time.time() - start)/60:.1f} minutes")
print(f"Selected {len(sfs_features)} features by SFS: {sfs_features}")

""" Sequential Backward Selection (SBS) (start from all -> remove worst features one by one) """
print("\nStarting Sequential Backward Selection (SBS) -> selecting 20 best features...")
start = time.time()
sbs = SFS(
    estimator=base_model_for_selection,  # using logistic regression as base model
    n_features_to_select=20,  # select top 20 features
    direction="backward",  # SBS
    scoring="accuracy",  # optimize for accuracy
    cv=3,  # 3-fold cross-validation (faster than 5-fold)
    n_jobs=-1,  # utilize all CPU cores
)
sbs.fit(x_scaled_df, y)  # fit SBS on scaled data
sbs_features = x.columns[sbs.get_support()].tolist()  # get selected feature names
print(f"SBS finished in {(time.time() - start)/60:.1f} minutes")
print(f"Selected {len(sbs_features)} features by SBS: {sbs_features}")

""" Bidirectional Feature Selection (start from 0 -> add best + remove worst iteratively) """
print("\nRunning Bidirectional selection (SFS -> then refine)...")
start = time.time()
bidir = SFS(
    estimator=base_model_for_selection,  # using logistic regression as base model
    n_features_to_select=20,  # select top 20 features
    direction="forward",  # Note: using 'forward' as sklearn doesn't have true bidirectional
    scoring="accuracy",  # optimize for accuracy
    cv=3,  # 3-fold cross-validation (faster than 5-fold)
    n_jobs=-1,  # utilize all CPU cores
)
bidir.fit(x_scaled_df, y)  # fit on scaled data
bidir_features = x.columns[bidir.get_support()].tolist()  # get selected feature names
print(f"Bidirectional finished in {(time.time() - start)/60:.1f} minutes")
print(f"Selected {len(bidir_features)} features by Bidirectional: {bidir_features}")


""" PCA Dimensionality Reduction (retain 95% variance) """
print("\nApplying PCA to retain 95% variance...")
pca = PCA(
    n_components=0.95,  # retain 95% variance
    svd_solver="full",  # use full SVD solver
    random_state=42,  # for reproducibility
)
x_pca = pca.fit_transform(x_scaled)  # fit PCA on scaled data
print(
    f"PCA reduced from {x.shape[1]} to {pca.n_components_} components"
)  # This shows number of components selected
print(
    f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}"
)  # This shows how much variance is retained

joblib.dump(pca, "pca_transformer.pkl")
print("Saved PCA transformer as 'pca_transformer.pkl'")

""" Save all processed datasets for modeling step """
print("\nSaving final feature sets as pickle files...")
pd.DataFrame(x_scaled_df[sfs_features]).to_pickle("x_sfs20_scaled.pkl")
pd.DataFrame(x_scaled_df[sbs_features]).to_pickle("x_sbs20_scaled.pkl")
pd.DataFrame(x_scaled_df[bidir_features]).to_pickle("x_bidir20_scaled.pkl")
pd.DataFrame(x_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)]).to_pickle(
    "x_pca95_scaled.pkl"
)
y.to_pickle("y_family.pkl")
print("All datasets saved successfully.")

print("\nFeature engineering completed.")

""" Final Comparison table & plot """
results = pd.DataFrame(
    {
        "Method": [
            "Original (raw)",
            "After duplicate removal",
            "SFS (20 features)",
            "SBS (20 features)",
            "Bidirectional (20 features)",
            "PCA (95% variance)",
        ],
        "Num Features/Components": [
            x_raw.shape[1],  # original feature count
            x.shape[1],  # after duplicate removal
            len(sfs_features),  # after SFS
            len(sbs_features),  # after SBS
            len(bidir_features),  # after Bidirectional
            pca.n_components_,  # after PCA
        ],
    }
)
print("\n" + "=" * 60)
print("TASK 3 – FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(results.to_string(index=False))
print("=" * 60)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Method",  # x-axis: feature engineering method
    y="Num Features/Components",  # y-axis: number of features/components
    hue="Method",  # color by method
    data=results,  # data source
    palette="viridis",  # color palette
    legend=False,  # no legend needed
)
plt.title(
    "Feature Engineering Methods vs. Number of Features/Components", fontsize=14, pad=20
)
plt.xticks(rotation=30, ha="right")  # align right
plt.ylabel("Number of Features/Components")
plt.tight_layout()  # adjust layout
plt.savefig(
    "feature_engineering_comparison.png", dpi=300, bbox_inches="tight"
)  # save figure
plt.show()
