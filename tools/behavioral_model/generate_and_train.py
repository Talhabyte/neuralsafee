"""
NeuralSafe — Behavioral Anomaly Classifier training script.

Trains a Random Forest + linear SVM ensemble to classify an 11-dim
z-score vector (the exact output of AnomalyEvaluator.evaluate() in the
Flutter app) into one of three tiers: Normal / Mildly Anomalous /
Critically Anomalous — matching NeuralSafe proposal Section 6.3.

DATA: No public labeled dataset of domestic-abuse-victim phone
behavior exists (for good reason — collecting one would require real
at-risk subjects and formal ethics review this FYP does not have).
This script instead generates a SYNTHETIC dataset of z-score vectors,
with class-conditional mean shifts chosen to reflect the isolation /
restricted-access signature described in the project's own literature
review (Lane et al. 2011; Saeb et al. 2015): reduced overall phone
activity and app variety, but more furtive/short, concentrated usage.
This is a documented methodological choice, not real behavioral data —
state this plainly in the thesis methodology and evaluation chapters.

Feature order matches lib/models/behavioral_baseline.dart's
`trackedFeatureKeys` exactly — this MUST stay in sync with that file.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Must match lib/models/behavioral_baseline.dart::trackedFeatureKeys order.
FEATURE_KEYS = [
    "screenSessionsPerHour",
    "meanScreenSessionMs",
    "medianScreenSessionMs",
    "shortSessionRatio",
    "userPresentCount",
    "appSessionsPerHour",
    "meanAppSessionMs",
    "medianAppSessionMs",
    "distinctPackagesCount",
    "totalAppUsageMs",
    "topPackageUsageRatio",
]
N_FEATURES = len(FEATURE_KEYS)
CLASSES = ["Normal", "MildlyAnomalous", "CriticallyAnomalous"]

# Per-class mean shift in z-score units. Sign convention: negative =
# below the user's own baseline, positive = above it. The "isolation /
# restricted access" signature used for the anomalous classes:
#   - fewer, shorter screen & app sessions (restricted free use)
#   - fewer distinct apps touched (narrowed access)
#   - MORE short/furtive sessions (shortSessionRatio) and usage
#     concentrated in fewer apps (topPackageUsageRatio) despite less
#     overall use — consistent with snatched, monitored phone checks.
MEAN_SHIFTS = {
    "Normal": np.zeros(N_FEATURES),
    "MildlyAnomalous": np.array([
        -1.2,   # screenSessionsPerHour
        -1.0,   # meanScreenSessionMs
        -1.0,   # medianScreenSessionMs
        +1.3,   # shortSessionRatio
        -0.8,   # userPresentCount
        -1.1,   # appSessionsPerHour
        -0.9,   # meanAppSessionMs
        -0.9,   # medianAppSessionMs
        -1.2,   # distinctPackagesCount
        -1.0,   # totalAppUsageMs
        +1.1,   # topPackageUsageRatio
    ]),
    "CriticallyAnomalous": np.array([
        -3.0, -2.6, -2.6, +3.2, -2.4, -2.9, -2.4, -2.4, -3.1, -2.7, +2.8,
    ]),
}

# Shared covariance: mild positive correlation between related features
# (e.g. meanScreenSessionMs / medianScreenSessionMs move together) plus
# per-sample noise, so classes overlap somewhat rather than being
# trivially separable — a flat 100% accuracy would not be a credible
# number to defend.
_base_std = np.ones(N_FEATURES)
_corr = np.eye(N_FEATURES)
_related_pairs = [(1, 2), (6, 7), (0, 5)]  # (mean,median) ms pairs; session rate pair
for i, j in _related_pairs:
    _corr[i, j] = _corr[j, i] = 0.6
COV = np.outer(_base_std, _base_std) * _corr

N_PER_CLASS = 3000


def make_dataset():
    X_parts, y_parts = [], []
    for label in CLASSES:
        mean = MEAN_SHIFTS[label]
        samples = rng.multivariate_normal(mean, COV, size=N_PER_CLASS)
        X_parts.append(samples)
        y_parts.append(np.full(N_PER_CLASS, label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def tree_to_dict(tree, feature_names, class_names):
    """Export a single sklearn DecisionTreeClassifier's internal tree_
    structure to a plain nested-dict JSON the Dart interpreter can walk.
    Leaf nodes carry a class probability vector (not just the argmax),
    so the Dart side can average probabilities across the forest."""
    t = tree.tree_

    def node_to_dict(i):
        if t.children_left[i] == t.children_right[i] == -1:
            counts = t.value[i][0]
            total = counts.sum()
            probs = (counts / total).tolist() if total > 0 else [0.0] * len(class_names)
            return {"leaf": True, "probs": probs}
        return {
            "leaf": False,
            "featureIndex": int(t.feature[i]),
            "threshold": float(t.threshold[i]),
            "left": node_to_dict(t.children_left[i]),
            "right": node_to_dict(t.children_right[i]),
        }

    return node_to_dict(0)


def export_random_forest(rf, feature_names, class_names):
    return {
        "type": "random_forest",
        "classes": class_names,
        "trees": [tree_to_dict(est, feature_names, class_names) for est in rf.estimators_],
    }


def export_linear_svm(svm, feature_names, class_names):
    # LinearSVC with multi_class='ovr': coef_ shape (n_classes, n_features),
    # intercept_ shape (n_classes,). Decision for class c is coef_[c]·x + intercept_[c].
    return {
        "type": "linear_svm",
        "classes": class_names,
        "coef": svm.coef_.tolist(),
        "intercept": svm.intercept_.tolist(),
    }


def softmax(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    out_dir = Path(__file__).parent
    X, y = make_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RNG_SEED
    )

    rf = RandomForestClassifier(
        n_estimators=40, max_depth=6, random_state=RNG_SEED, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    svm = LinearSVC(
        multi_class="ovr", C=1.0, max_iter=5000, random_state=RNG_SEED, class_weight="balanced"
    )
    svm.fit(X_train, y_train)

    # --- Evaluate each model individually ---
    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)

    print("=== Random Forest ===")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(classification_report(y_test, rf_pred, labels=CLASSES))

    print("=== Linear SVM ===")
    print(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
    print(classification_report(y_test, svm_pred, labels=CLASSES))

    # --- Ensemble: average RF's predict_proba with softmax(SVM decision_function) ---
    rf_proba = rf.predict_proba(X_test)
    # Ensure column order matches CLASSES regardless of sklearn's internal class ordering
    rf_class_order = list(rf.classes_)
    rf_proba = rf_proba[:, [rf_class_order.index(c) for c in CLASSES]]

    svm_scores = svm.decision_function(X_test)  # shape (n_samples, n_classes) for 3-class OVR
    svm_class_order = list(svm.classes_)
    svm_scores = svm_scores[:, [svm_class_order.index(c) for c in CLASSES]]
    svm_proba = softmax(svm_scores)

    ensemble_proba = 0.5 * rf_proba + 0.5 * svm_proba
    ensemble_pred = np.array(CLASSES)[np.argmax(ensemble_proba, axis=1)]

    print("=== Ensemble (0.5 RF + 0.5 SVM) ===")
    acc = accuracy_score(y_test, ensemble_pred)
    print(f"Accuracy: {acc:.4f}")
    report = classification_report(y_test, ensemble_pred, labels=CLASSES)
    print(report)
    cm = confusion_matrix(y_test, ensemble_pred, labels=CLASSES)
    print("Confusion matrix (rows=true, cols=predicted), order:", CLASSES)
    print(cm)

    # --- Save evaluation artifacts for the thesis evaluation chapter ---
    with open(out_dir / "evaluation_report.txt", "w") as f:
        f.write("NeuralSafe Behavioral Module — Ensemble Classifier Evaluation\n")
        f.write(f"(synthetic dataset, N={N_PER_CLASS} per class, seed={RNG_SEED})\n\n")
        f.write(f"Random Forest accuracy: {accuracy_score(y_test, rf_pred):.4f}\n")
        f.write(f"Linear SVM accuracy:    {accuracy_score(y_test, svm_pred):.4f}\n")
        f.write(f"Ensemble accuracy:      {acc:.4f}\n\n")
        f.write("Ensemble classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=predicted), order: " + str(CLASSES) + "\n")
        f.write(str(cm))

    # --- Export models to JSON for the Dart interpreter ---
    model_export = {
        "featureKeys": FEATURE_KEYS,
        "classes": CLASSES,
        "ensembleWeights": {"randomForest": 0.5, "linearSvm": 0.5},
        "scoreBandMidpoints": {"Normal": 10.0, "MildlyAnomalous": 38.0, "CriticallyAnomalous": 78.0},
        "randomForest": export_random_forest(rf, FEATURE_KEYS, CLASSES),
        "linearSvm": export_linear_svm(svm, FEATURE_KEYS, CLASSES),
    }
    with open(out_dir / "behavioral_classifier_model.json", "w") as f:
        json.dump(model_export, f)

    size_kb = (out_dir / "behavioral_classifier_model.json").stat().st_size / 1024
    print(f"\nExported model JSON: {size_kb:.1f} KB")
    print(f"Evaluation report written to evaluation_report.txt")


if __name__ == "__main__":
    main()
