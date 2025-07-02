from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from preprocessing_pipeline import load_prepared_data

MODEL_INPUT_PATH = "rf_multioutput_pipeline.joblib"
FINAL_MODEL_PATH = "final_rf_model.joblib"
PREDICTIONS_CSV = "rf_predictions.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def compute_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> None:
    """Print RMSE and Pearson r for each target column."""
    target_cols = y_true.columns
    for i, col in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
        r, _ = pearsonr(y_true.iloc[:, i], y_pred[:, i])
        print(f"{col: >25} — RMSE: {rmse:.4f},  Pearson r: {r:.4f}")


def main():
    print("Loading prepared data …")
    X, y = load_prepared_data()

    print("Splitting into train/test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")  # type: ignore[attr-defined]

    print(f"Loading best estimator from '{MODEL_INPUT_PATH}' …")
    try:
        best_model = joblib.load(MODEL_INPUT_PATH)
    except FileNotFoundError as err:
        raise SystemExit(
            f"Model file '{MODEL_INPUT_PATH}' not found. Run hyperparameter_search.py first."
        ) from err

    # Re-fit model to training data to avoid leakage (preprocessor is inside the pipeline)
    print("Fitting best model on training data …")
    best_model.fit(X_train, y_train)

    print("Making predictions on test data …")
    preds = best_model.predict(X_test)

    print("Calculating metrics …")
    compute_metrics(y_test, preds)  # type: ignore[arg-type]

    # Build prediction DataFrame with metadata columns
    pred_df = pd.DataFrame(preds, columns=y.columns, index=X_test.index)  # type: ignore[arg-type]
    # Add CID and subject columns
    pred_df.insert(0, "subject #", X_test.loc[:, "subject #"])  # type: ignore[index]
    pred_df.insert(0, "CID", X_test.loc[:, "CID"])  # type: ignore[index]

    print(f"Saving predictions → '{PREDICTIONS_CSV}'.")
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    print(f"Saving final trained model → '{FINAL_MODEL_PATH}'.")
    joblib.dump(best_model, FINAL_MODEL_PATH)

    print("Done.")


if __name__ == "__main__":
    main() 