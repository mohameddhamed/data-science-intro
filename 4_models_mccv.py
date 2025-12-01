# ============================= TASK 4 + TASK 5: MODELS + MONTE-CARLO CV (ONE FILE) =============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

""" Load the best feature set and labels for modeling """
print("Loading feature set and labels...")
X = pd.read_pickle("X_sfs20_scaled.pkl")
y = pd.read_pickle("y.pkl")

print(f"Using feature set with {X.shape[1]} features → {X.shape[0]:,} samples")

# --------------------------------------------------------------------------------------------------
# 2. DEFINE THE THREE MODELS
# --------------------------------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=20
    ),  # tuned max_depth
    "Gaussian Naive Bayes": GaussianNB(),  # default params
    "k-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1
    ),  # tuned n_neighbors
}

# --------------------------------------------------------------------------------------------------
# 3. MONTE-CARLO CROSS-VALIDATION (200 iterations)
# --------------------------------------------------------------------------------------------------
n_iterations = 200  # number of Monte-Carlo iterations
results = []

print(
    f"\nStarting Monte-Carlo CV ({n_iterations} iterations) — this takes ~3–5 minutes...\n"
)
print("-" * 90)

for name, model in models.items():  # loop over models
    print(f"Running {name}...")
    acc_list, f1_list, recall_dict = (
        [],
        [],
        {label: [] for label in y.unique()},
    )  # to store per-class recall

    for i in range(n_iterations):  # Monte-Carlo iterations
        if (i + 1) % 50 == 0:  # print progress every 50 iterations
            print(f"   → iteration {i+1}/{n_iterations}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=i
        )  # stratified split

        model.fit(X_train, y_train)  # train
        y_pred = model.predict(X_test)  # predict

        # Metrics
        acc_list.append(accuracy_score(y_test, y_pred))  # accuracy
        f1_list.append(f1_score(y_test, y_pred, average="macro"))  # macro F1

        # Per-class recall
        recall_per_class = recall_score(
            y_test, y_pred, average=None, labels=y.unique()
        )  # in label order
        for label, rec in zip(y.unique(), recall_per_class):  # map label to recall
            recall_dict[label].append(rec)  # store recall for this class

    # Compute mean ± std
    results.append(
        {  # store results for this model
            "Model": name,  # model name
            "Accuracy": f"{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}",  # mean ± std accuracy
            "Macro-F1": f"{np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}",  # mean ± std macro F1
        }
    )

    # Add per-class recall (mean only)
    for label in y.unique():  # loop over classes
        mean_rec = np.mean(recall_dict[label])  # mean recall for this class
        results[-1][f"Recall_{label}"] = f"{mean_rec:.4f}"  # store in results

""" Final results table """
df_results = pd.DataFrame(results)  # create dataframe from results
df_results = df_results.set_index("Model")  # set model name as index

print("\n" + "=" * 100)
print("TASK 4 + 5 FINAL RESULTS (Monte-Carlo CV – 200 iterations)")
print("=" * 100)
print(df_results.to_string())
print("=" * 100)

# Save table as image for report
plt.figure(figsize=(14, 6))
sns.heatmap(
    df_results.astype(float).round(4),
    annot=True,
    cmap="RdYlGn",
    fmt="s",
    linewidths=0.5,
)  # heatmap
plt.title("Model Comparison – Monte-Carlo CV (200 iterations)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig("model_comparison_mccv.png", dpi=300, bbox_inches="tight")
plt.show()

""" Stability plot across iterations"""
print("\nGenerating stability plot...")
all_acc = {}  # to store accuracies per model

for name, model in models.items():
    acc_list = []  # to store accuracies for this model
    for i in range(n_iterations):  # Monte-Carlo iterations
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=i
        )  # stratified split
        model.fit(X_train, y_train)  # train
        y_pred = model.predict(X_test)  # predict
        acc_list.append(accuracy_score(y_test, y_pred))  # accuracy
    all_acc[name] = acc_list

plt.figure(figsize=(12, 7))
for name, accs in all_acc.items():
    plt.plot(accs, label=f"{name} (mean={np.mean(accs):.4f})", alpha=0.8)

plt.title("Model Accuracy Stability Across 200 Monte-Carlo Splits", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mccv_stability_plot.png", dpi=300)
plt.show()
