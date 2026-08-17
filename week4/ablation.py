import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from extraction import main as extract_all

K_NEIGHBORS = 3
TEST_SIZE = 0.3
RANDOM_STATE = 42

FEATURE_ORDER = ["mean", "std", "skew", "delta_var"]

ABLATION_SETS = {
    "1. Raw RSSI avg only": ["mean"],
    "2. Avg + Rolling Std": ["mean", "std"],
    "3. Full (Avg+Std+Skew+DeltaVar)": ["mean", "std", "skew", "delta_var"],
}


def evaluate_set(all_features, feature_keys, k=K_NEIGHBORS):
    X = [[f[key] for key in feature_keys] for f in all_features]
    y = [f["label"] for f in all_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    k_eff = min(k, len(X_train))
    clf = KNeighborsClassifier(n_neighbors=k_eff).fit(X_train, y_train)
    preds = clf.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds, average="macro", zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, average="macro", zero_division=0), 3),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def main():
    print("=" * 60)
    print("Step 1: Extracting features from all labeled recordings")
    print("=" * 60)
    all_features = extract_all()

    if len(all_features) < 10:
        print("\nWARNING: Very few total samples. Results will be noisy.")

    print("\n" + "=" * 60)
    print("Step 2: Running ablation comparison")
    print("=" * 60)

    results = {}
    for name, feature_keys in ABLATION_SETS.items():
        results[name] = evaluate_set(all_features, feature_keys)

    print(f"\n{'Feature Set':<35} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'n_test':>7}")
    print("-" * 72)
    for name, r in results.items():
        print(
            f"{name:<35} {r['accuracy']:>9} {r['precision']:>10} {r['recall']:>8} {r['n_test']:>7}"
        )

    with open("ablation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature_set", "accuracy", "precision", "recall", "n_train", "n_test"]
        )
        writer.writeheader()
        for name, r in results.items():
            writer.writerow({"feature_set": name, **r})

    print("\nSaved ablation_results.csv")
    return results


if __name__ == "__main__":
    main()
