import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import scipy.stats as stats

""" Load the best feature set and labels from Task 3 """
print("Loading best feature set from Task 3...")
X = pd.read_pickle("X_sfs20_scaled.pkl")  # ← this gave the highest scores in Task 4+5
y = pd.read_pickle("y.pkl")
print(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")

# base model for comparison
base_dt = DecisionTreeClassifier(
    random_state=42, max_depth=20
)  # tuned max_depth from Task 4+5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)  # stratified split

print("\nTraining base Decision Tree...")
base_dt.fit(X_train, y_train)  # train base model
y_pred_base = base_dt.predict(X_test)

base_acc = accuracy_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base, average="macro")  # macro F1
print(f"Base model → Accuracy: {base_acc:.5f} | Macro-F1: {base_f1:.5f}")

""" Hyperparameter Tuning with RandomizedSearchCV """
print("\nStarting RandomizedSearchCV (100 iterations) — this takes ~4–7 minutes...")

param_dist = {
    "max_depth": stats.randint(10, 50),  # 10–50
    "min_samples_split": stats.randint(2, 20),  # for splitting a node
    "min_samples_leaf": stats.randint(1, 20),  # for leaf nodes
    "max_features": stats.uniform(
        0.3, 0.7
    ),  # 30%–100% of features, serves to reduce overfitting
    "criterion": ["gini", "entropy"],  # splitting criteria
    "class_weight": [None, "balanced"],  # handle class imbalance
}

random_search = RandomizedSearchCV(  # randomized search
    estimator=DecisionTreeClassifier(random_state=42),  # base model
    param_distributions=param_dist,  # parameter distributions
    n_iter=100,  # 100 random combinations (perfect balance speed/quality)
    scoring="f1_macro",  # optimize for macro F1
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # utilize all CPU cores
    random_state=42,  # for reproducibility
    verbose=1,  # show progress
)

random_search.fit(X_train, y_train)

# Retrieve and evaluate the best model
best_dt = random_search.best_estimator_  # best model from random search
print("\nTUNING COMPLETE!")
print("Best parameters found:")
for param, value in random_search.best_params_.items():  # print each best param
    print(f"  {param}: {value}")

y_pred_tuned = best_dt.predict(X_test)  # predict with tuned model

tuned_acc = accuracy_score(y_test, y_pred_tuned)  # accuracy
tuned_f1 = f1_score(y_test, y_pred_tuned, average="macro")  # macro F1

print(f"\nBase model   → Accuracy: {base_acc:.5f} | Macro-F1: {base_f1:.5f}")
print(f"Tuned model  → Accuracy: {tuned_acc:.5f} | Macro-F1: {tuned_f1:.5f}")
print(
    f"Improvement  → Accuracy +{(tuned_acc-base_acc)*100:.3f}% | F1 +{(tuned_f1-base_f1)*100:.3f}%"
)

comparison = pd.DataFrame(
    {
        "Model": ["Base Decision Tree", "Tuned Decision Tree"],
        "Accuracy": [base_acc, tuned_acc],
        "Macro-F1": [base_f1, tuned_f1],
    }
)

print("\n" + "=" * 80)
print("TASK 6 – HYPERPARAMETER TUNING RESULTS")
print("=" * 80)
print(comparison.round(5))
print("=" * 80)

# Bar plot
plt.figure(figsize=(10, 6))
comparison_melted = comparison.melt(  # long format for seaborn
    id_vars="Model", var_name="Metric", value_name="Score"
)
sns.barplot(
    x="Model", y="Score", hue="Metric", data=comparison_melted, palette="Set2"
)  # bar plot
plt.title("Base vs Tuned Decision Tree Performance", fontsize=16, pad=20)  # title
plt.ylim(0.90, 0.98)  # y-axis limits for better visibility
for i, row in comparison_melted.iterrows():  # add text labels
    plt.text(
        i % 2, row["Score"] + 0.001, f"{row['Score']:.5f}", ha="center", fontsize=10
    )
plt.tight_layout()
plt.savefig("tuning_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
