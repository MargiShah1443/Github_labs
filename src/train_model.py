# src/train_model.py (full file not required; paste this whole script if easier)
import os, argparse, datetime, json, inspect
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import mlflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    ts = args.timestamp

    # --- data ---
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- model (Gradient Boosting) + Platt calibration ---
    base = GradientBoostingClassifier(random_state=42)

    # handle sklearn versions: use 'estimator' if available, else 'base_estimator'
    sig = inspect.signature(CalibratedClassifierCV.__init__)
    if "estimator" in sig.parameters:
        calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    else:
        calibrated = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)

    model = calibrated.fit(X_train, y_train)

    # --- metrics ---
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "F1_Score": float(f1_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall": float(recall_score(y_test, y_pred)),
        "ROC_AUC": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X.shape[1]),
    }

    # --- log locally with mlflow (optional for grading, safe to keep) ---
    mlflow.set_tracking_uri("./mlruns")
    exp_name = f"BreastCancer_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    exp_id = mlflow.create_experiment(exp_name)
    with mlflow.start_run(experiment_id=exp_id, run_name="GB-Calibrated"):
        mlflow.log_params({"dataset": "sklearn_breast_cancer", "model": "GB + Platt"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    # --- save artifacts (repo root; workflow moves them) ---
    model_filename = f"model_{ts}_gb_calibrated.joblib"
    dump(model, model_filename)

    os.makedirs("metrics", exist_ok=True)
    with open(f"metrics/{ts}_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved {model_filename} and metrics/{ts}_train_metrics.json")
