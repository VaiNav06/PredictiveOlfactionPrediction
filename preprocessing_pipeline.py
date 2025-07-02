from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

# Load pre-saved data produced by data_setup.py, or recompute if necessary.
# We expect X and y to exist in memory if this is imported from another script.
# For standalone execution, we read and prepare the data inline.

def load_prepared_data(
    descriptors_path: str = "MolecularDescriptorData.txt",
    trainset_path: str = "TrainSetTabSeperated.txt",
):
    """Return X, y ready for fitting the ColumnTransformer.

    Mirrors the logic from data_setup.py so this script can be run independently.
    """
    df_desc = pd.read_csv(descriptors_path, sep="\t")
    df_train = pd.read_csv(trainset_path, sep="\t")
    master_df = pd.merge(df_train, df_desc, left_on ="Compound Identifier", right_on="CID", how="inner")

    # Ensure no missing values remain anywhere in the dataset
    master_df = master_df.fillna(0)

    meta_cols = {"Compound Identifier", "CID", "subject #", "Dilution"}

    # Identify target columns: numeric columns in train set excluding metadata
    numeric_cols = df_train.select_dtypes(include=["number"]).columns
    target_cols = [c for c in numeric_cols if c not in meta_cols]

    # Drop any other non-useful columns from the merged df
    irrelevant_cols = [c for c in df_train.columns if c not in meta_cols and c not in target_cols]
    if irrelevant_cols:
        master_df = master_df.drop(columns=irrelevant_cols)

    X = master_df.drop(columns=target_cols)
    y = master_df[target_cols]
    return X, y


def build_column_transformer(
    X: pd.DataFrame,
    alpha: float = 0.01,
    random_state: int | None = 42,
) -> ColumnTransformer:
    """Construct the ColumnTransformer with the desired preprocessing steps."""
    categorical_features = ["subject #", "Dilution"]

    # Pipeline for categorical columns: impute missing with 0 then one-hot encode
    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    # Identify descriptor columns (exclude categorical + id cols)
    descriptor_features = [
        col
        for col in X.columns
        if col not in categorical_features + ["CID", "Compound Identifier"]
    ]

    desc_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
            (
                "select",
                SelectFromModel(
                    estimator=Lasso(alpha=alpha, random_state=random_state),
                    max_features=None,  # Let SelectFromModel decide
                    threshold="mean",  # Select features with importance > mean
                ),
            ),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                cat_pipeline,
                categorical_features,
            ),
            ("desc", desc_pipeline, descriptor_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


if __name__ == "__main__":
    print("Loading data …")
    X, y = load_prepared_data()

    print("Building ColumnTransformer …")
    preprocessor = build_column_transformer(X)

    print("Fitting transformer to data (this may take a moment) …")
    preprocessor.fit(X, y)

    print("Transformation complete. Saving fitted transformer → 'preprocessor.joblib'.")
    joblib.dump(preprocessor, "preprocessor.joblib")

    # Demonstrate transformation shape
    X_trans = preprocessor.transform(X)
    if hasattr(X_trans, "shape"):
        print("Transformed X shape:", X_trans.shape)
    else:
        # csr_matrix shape
        from scipy.sparse import issparse

        if issparse(X_trans):  # pragma: no cover
            print("Transformed X shape:", X_trans.shape) 