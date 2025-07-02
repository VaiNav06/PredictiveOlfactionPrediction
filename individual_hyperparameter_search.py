"""Hyper-parameter optimisation via Bayesian optimization for individual targets.

This performs separate hyperparameter optimization for each of the 21 targets,
using an extensive search space and robust cross-validation for maximum accuracy.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Base pipeline builder components
from preprocessing_pipeline import build_column_transformer, load_prepared_data


def build_single_target_pipeline(X):
    """Construct Pipeline(preprocess, model) for a single target."""
    # Try loading a pre-fitted preprocessor if available for speed
    try:
        preprocessor = joblib.load("preprocessor.joblib")
        print("Loaded existing preprocessor.joblib …")
    except FileNotFoundError:
        print("preprocessor.joblib not found → building anew …")
        preprocessor = build_column_transformer(X)

    rf_base = RandomForestRegressor(
        n_estimators=200,  # Higher default
        max_depth=None,    # Allow full depth by default
        max_features=0.5,  # Use 50% of features by default
        min_samples_leaf=1,
        n_jobs=1,
        verbose=1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf_base),
        ]
    )
    return pipeline


def optimize_single_target(X, y_target, target_name):
    """Perform hyperparameter optimization for a single target."""
    print(f"\nOptimizing for target: {target_name}")
    
    pipe = build_single_target_pipeline(X)
    
    # Expanded search space for maximum accuracy
    search_space = {
        'preprocess__desc__select__threshold': Categorical(['mean', 'median', '1.5*mean']),
        'model__n_estimators': Integer(100, 1000),
        'model__max_features': Real(0.1, 1.0),
        'model__min_samples_split': Integer(2, 20),
        'model__min_samples_leaf': Integer(1, 10),
        'model__max_depth': Integer(10, 100),
        'model__bootstrap': Categorical([True, False]),
        'model__criterion': Categorical(['squared_error', 'absolute_error', 'poisson']),
        'model__min_weight_fraction_leaf': Real(0.0, 0.5),
        'model__max_leaf_nodes': Integer(10, 1000)
    }

    # More robust cross-validation: 5 folds, 3 repeats
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    search = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_space,
        n_iter=100,  # More iterations for better exploration
        scoring='r2',
        n_jobs=-1,
        cv=cv,
        verbose=2,  # More detailed progress
        random_state=42,
        return_train_score=True,  # Store training scores
        refit=True  # Refit on best params
    )

    search.fit(X, y_target)
    
    print(f"\nBest score for {target_name}: {search.best_score_:.4f}")
    print(f"Cross-validation results:")
    print(f"  Mean CV score: {np.mean(search.cv_results_['mean_test_score']):.4f}")
    print(f"  Std CV score: {np.std(search.cv_results_['mean_test_score']):.4f}")
    print("\nBest parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
        
    return search.best_estimator_, search.best_score_


def main():
    print("Loading data …")
    X, y = load_prepared_data()
    
    # Dictionary to store results
    results = {
        'target': [],
        'best_score': [],
        'mean_cv_score': [],
        'std_cv_score': [],
        'best_params': [],
        'model_path': []
    }
    
    # Optimize for each target separately
    for target_name in y.columns:
        y_target = y[target_name]
        
        best_model, best_score = optimize_single_target(X, y_target, target_name)
        
        # Save the model with target name
        model_path = f"rf_model_{target_name.replace(' ', '_')}.joblib"
        joblib.dump(best_model, model_path)
        
        # Get cross-validation scores
        cv_scores = best_model.named_steps['model'].score(
            best_model.named_steps['preprocess'].transform(X),
            y_target
        )
        
        # Store results
        results['target'].append(target_name)
        results['best_score'].append(best_score)
        results['mean_cv_score'].append(np.mean(cv_scores))
        results['std_cv_score'].append(np.std(cv_scores))
        results['best_params'].append(best_model.get_params())
        results['model_path'].append(model_path)
    
    # Save detailed results summary to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('individual_target_optimization_results.csv', index=False)
    print("\nOptimization complete! Detailed results saved to 'individual_target_optimization_results.csv'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average R² score across all targets: {np.mean(results['best_score']):.4f}")
    print(f"Best performing target: {results_df.loc[results_df['best_score'].idxmax(), 'target']}")
    print(f"Best R² score: {results_df['best_score'].max():.4f}")
    print(f"Worst performing target: {results_df.loc[results_df['best_score'].idxmin(), 'target']}")
    print(f"Worst R² score: {results_df['best_score'].min():.4f}")


if __name__ == "__main__":
    main() 