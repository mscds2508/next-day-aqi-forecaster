"""
evaluate.py
-----------
Evaluates a trained model on the held-out test set and saves
metrics + an actual-vs-predicted plot.
"""

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src import config


def rmse(y_true, y_pred) -> float:
    """Compute RMSE in a way that works across sklearn versions
    (newer sklearn removed the `squared` argument from mean_squared_error)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(pipeline=None, X_test=None, y_test=None):
    if pipeline is None:
        pipeline = joblib.load(config.MODEL_PATH)

    if X_test is None or y_test is None:
        from src.data_loader import load_clean_data
        from src.feature_engineering import build_features
        from src.train import get_train_test_split

        df = load_clean_data()
        df = build_features(df)
        _, X_test, _, y_test = get_train_test_split(df)

    preds = pipeline.predict(X_test)

    metrics = {
        "RMSE": rmse(y_test, preds),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
    }

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Actual vs Predicted plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.3, s=10)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Actual vs Predicted AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PRED_PLOT_PATH, dpi=150)
    print(f"Saved plot to {config.PRED_PLOT_PATH}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
