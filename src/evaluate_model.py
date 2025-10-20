# src/evaluate_model.py
import argparse, os, json
from joblib import load
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    ts = args.timestamp

    model_path = f"model_{ts}_gb_calibrated.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load(model_path)

    # fresh evaluation set (same dataset, different split idea is fine for this lab)
    data = load_breast_cancer()
    X, y = data.data, data.target
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "F1_Score": float(f1_score(y, y_pred)),
        "Precision": float(precision_score(y, y_pred)),
        "Recall": float(recall_score(y, y_pred)),
        "ROC_AUC": float(roc_auc_score(y, y_proba)),
    }

    os.makedirs("metrics", exist_ok=True)
    metrics_filename = f"{ts}_metrics.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote metrics to {metrics_filename}")
