from __future__ import annotations


from sklearn.exceptions import ConvergenceWarning 
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np

from preprocessing_pipeline import build_column_transformer, load_prepared_data


DEFAULT_MODEL_PATH = "rf_multioutput_pipeline.joblib"


def build_full_pipeline(X, batch_size=5):
    """Construct Pipeline(preprocess, model) with batch processing."""
    # Try loading a pre-fitted preprocessor if available for speed
    try:
        preprocessor = joblib.load("preprocessor.joblib")
        print("Loaded existing preprocessor.joblib …")
    except FileNotFoundError:
        print("preprocessor.joblib not found → building anew …")
        preprocessor = build_column_transformer(X)

    rf_base = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        max_features=0.5,
        min_samples_leaf=5,
        n_jobs=1,
        verbose=1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", MultiOutputRegressor(rf_base, n_jobs=-1)),
        ]
    )
    return pipeline


def main():
    print("Loading prepared data …")
    X, y = load_prepared_data()

    print("Building full pipeline …")
    pipe = build_full_pipeline(X)

    print("Fitting full pipeline (may take some time) …")
    pipe.fit(X, y)

    print(f"Saving fitted pipeline → '{DEFAULT_MODEL_PATH}'.")
    joblib.dump(pipe, DEFAULT_MODEL_PATH)


if __name__ == "__main__":
    main() 