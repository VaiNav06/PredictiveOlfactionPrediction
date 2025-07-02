from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import joblib

# Constants
TRAIN_DATA_PATH = "TrainSetTabSeperated.txt"
MOLECULAR_DESCRIPTORS_PATH = "MolecularDescriptorData.txt"
N_SPLITS = 5
RANDOM_STATE = 42
POPULATION_MODELS_OUTPUT_PATH = "new_population_models.joblib"

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and molecular descriptor data."""
    print("\nLoading data...")
    
    # Load training data
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep="\t")
    
    # Load molecular descriptors
    mol_df = pd.read_csv(MOLECULAR_DESCRIPTORS_PATH, sep="\t")
    mol_df.columns = mol_df.columns.str.strip()
    mol_df = mol_df.rename(columns={'CID': 'Compound Identifier'})
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Molecular descriptors shape: {mol_df.shape}")
    return train_df, mol_df

def prepare_population_statistics(train_df: pd.DataFrame, mol_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], StandardScaler]:
    """Prepare features and population-level statistics."""
    print("\nPreparing features and population statistics...")
    
    # Get descriptor columns
    descriptor_cols = [
        'INTENSITY/STRENGTH', 'VALENCE/PLEASANTNESS', 'BAKERY', 'SWEET', 'FRUIT',
        'FISH', 'GARLIC', 'SPICES', 'COLD', 'SOUR', 'BURNT', 'ACID', 'WARM',
        'MUSKY', 'SWEATY', 'AMMONIA/URINOUS', 'DECAYED', 'WOOD', 'GRASS',
        'FLOWER', 'CHEMICAL'
    ]
    
    # Clean dilution values
    train_df['Dilution'] = train_df['Dilution'].str.strip()
    
    # Calculate population statistics for each compound-dilution combination
    population_stats = []
    # Ensure unique_compounds is a numpy array
    unique_compounds = train_df['Compound Identifier'].dropna().unique()
    
    print("Calculating population statistics...")
    for compound in unique_compounds:
        compound_data = train_df[train_df['Compound Identifier'] == compound]
        # Ensure unique dilutions is a numpy array
        for dilution in compound_data['Dilution'].dropna().unique():
            subset = compound_data[compound_data['Dilution'] == dilution]
            
            # Calculate mean and std for each descriptor
            stats = {}
            for desc in descriptor_cols:
                # Ensure values is a numpy array of floats
                values = subset[desc].dropna().values.astype(float)
                values = values[~np.isnan(values)]  # Remove NaN values
                if len(values) > 0:
                    stats.update({
                        f'{desc}_mean': np.mean(values),
                        f'{desc}_std': np.std(values) if len(values) > 1 else 0
                    })
                else:
                    stats.update({
                        f'{desc}_mean': 0,
                        f'{desc}_std': 0
                    })
            
            stats['Compound Identifier'] = compound
            stats['Dilution'] = dilution
            population_stats.append(stats)
    
    # Convert to DataFrame
    pop_stats_df = pd.DataFrame(population_stats)
    
    # Merge with molecular descriptors
    df = pd.merge(pop_stats_df, mol_df, on='Compound Identifier', how='inner')
    
    # Create feature matrix
    molecular_descriptor_cols = [col for col in mol_df.columns if col != 'Compound Identifier']
    
    # One-hot encode dilution
    df_encoded = pd.get_dummies(df[['Dilution']], prefix='dilution')
    
    # Combine features
    # Ensure feature_names is a list of strings
    feature_names: List[str] = df_encoded.columns.tolist() + molecular_descriptor_cols
    X = np.hstack([
        df_encoded.values.astype(float),         # One-hot encoded dilution
        df[molecular_descriptor_cols].values.astype(float)  # Molecular descriptors
    ])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prepare target variables (means and stds for each descriptor)
    target_cols = []
    y_means = []
    y_stds = []
    
    for desc in descriptor_cols:
        mean_col = f'{desc}_mean'
        std_col = f'{desc}_std'
        target_cols.append((mean_col, std_col))
        y_means.append(df[mean_col].values)
        y_stds.append(df[std_col].values)
    
    y_means = np.column_stack(y_means)
    y_stds = np.column_stack(y_stds)
    
    # Get groups for cross-validation (compound IDs)
    groups: np.ndarray = df['Compound Identifier'].values.astype(str) # Explicitly cast to numpy array of strings
    
    # Print feature information
    print(f"Number of features: {X_scaled.shape[1]}")
    print("Feature composition:")
    print(f"  - Dilution one-hot: {df_encoded.shape[1]} features")
    print(f"  - Molecular descriptors: {len(molecular_descriptor_cols)} features")
    
    # Print target information
    print("\nTarget variable statistics:")
    for i, desc in enumerate(descriptor_cols):
        print(f"\n{desc}:")
        print("  Means:")
        print(f"    - Mean of means: {np.mean(y_means[:, i]):.2f}")
        print(f"    - Std of means: {np.std(y_means[:, i]):.2f}")
        print("  Standard Deviations:")
        print(f"    - Mean of stds: {np.mean(y_stds[:, i]):.2f}")
        print(f"    - Std of stds: {np.std(y_stds[:, i]):.2f}")
    
    return X_scaled, y_means, y_stds, groups, descriptor_cols, feature_names, scaler

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, descriptor_cols: List[str], stat_type: str) -> Dict[str, Dict[str, float]]:
    """Calculate evaluation metrics for predictions."""
    results = {}
    for i, desc in enumerate(descriptor_cols):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # Remove NaN values
        mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
        true_vals_clean = true_vals[mask]
        pred_vals_clean = pred_vals[mask]
        
        if len(true_vals_clean) == 0:
            results[desc] = {'RMSE': float('nan'), 'R²': float('nan'), 'Correlation': float('nan')}
            continue
        
        rmse = np.sqrt(mean_squared_error(true_vals_clean, pred_vals_clean))
        r2 = r2_score(true_vals_clean, pred_vals_clean)
        correlation = pearsonr(true_vals_clean, pred_vals_clean)[0]
        
        results[desc] = {
            'RMSE': rmse,
            'R²': r2,
            'Correlation': correlation
        }
    return results

def analyze_feature_importance(model, feature_names: List[str], descriptor_cols: List[str], stat_type: str):
    """Analyze feature importance."""
    print(f"\nFeature Importance Analysis for {stat_type}:")
    print("=" * 50)
    
    for i, desc in enumerate(descriptor_cols):
        # Get the random forest regressor for this descriptor
        rf_model = model.estimators_[i]
        
        # Get feature importance
        importance = rf_model.feature_importances_
        
        # Sort features by importance
        feature_imp = list(zip(feature_names, importance))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{desc} - Top 10 Most Important Features:")
        for feat, imp in feature_imp[:10]:
            print(f"  {feat}: {imp:.4f}")

def cross_validate():
    """Perform compound-based cross-validation."""
    # Load data
    train_df, mol_df = load_data()
    X, y_means, y_stds, groups, descriptor_cols, feature_names, scaler = prepare_population_statistics(train_df, mol_df)
    
    # Initialize cross-validation
    cv = GroupKFold(n_splits=N_SPLITS)
    
    # Store results for each fold
    mean_fold_results = []
    std_fold_results = []
    
    print(f"\nPerforming {N_SPLITS}-fold cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_means, groups)):
        print(f"\nFold {fold_idx + 1}/{N_SPLITS}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_means_train, y_means_val = y_means[train_idx], y_means[val_idx]
        y_stds_train, y_stds_val = y_stds[train_idx], y_stds[val_idx]
        
        # Train models for means
        mean_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        print("Training mean prediction model...")
        mean_model.fit(X_train, y_means_train)
        
        # Train models for standard deviations
        std_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        print("Training standard deviation prediction model...")
        std_model.fit(X_train, y_stds_train)
        
        # Make predictions
        print("Making predictions...")
        y_means_pred = mean_model.predict(X_val)
        y_stds_pred = std_model.predict(X_val)
        
        # Evaluate
        mean_fold_result = evaluate_predictions(y_means_val, y_means_pred, descriptor_cols, "means")
        std_fold_result = evaluate_predictions(y_stds_val, y_stds_pred, descriptor_cols, "standard deviations")
        
        mean_fold_results.append(mean_fold_result)
        std_fold_results.append(std_fold_result)
        
        # Print fold results
        print("\nFold Results:")
        print("=" * 50)
        
        print("\nMean Predictions:")
        for desc in descriptor_cols:
            metrics = mean_fold_result[desc]
            print(f"\n{desc}:")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  R²: {metrics['R²']:.2f}")
            print(f"  Correlation: {metrics['Correlation']:.2f}")
        
        print("\nStandard Deviation Predictions:")
        for desc in descriptor_cols:
            metrics = std_fold_result[desc]
            print(f"\n{desc}:")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  R²: {metrics['R²']:.2f}")
            print(f"  Correlation: {metrics['Correlation']:.2f}")
        
        # Analyze feature importance
        analyze_feature_importance(mean_model, feature_names, descriptor_cols, "means")
        analyze_feature_importance(std_model, feature_names, descriptor_cols, "standard deviations")
    
    # Train final models on all data
    print("\nTraining final models on all data...")
    
    final_mean_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    final_std_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    print("Training final mean prediction model...")
    final_mean_model.fit(X, y_means)
    
    print("Training final standard deviation prediction model...")
    final_std_model.fit(X, y_stds)
    
    # Save the models and feature names
    print("\nSaving models and feature information...")
    model_data = {
        'mean_model': final_mean_model,
        'std_model': final_std_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'descriptor_cols': descriptor_cols
    }
    joblib.dump(model_data, POPULATION_MODELS_OUTPUT_PATH)
    print(f"Final population models, scaler, and feature names saved to {POPULATION_MODELS_OUTPUT_PATH}")

def main():
    print("Loading data for population model training...")
    train_df, mol_df = load_data()
    X, y_means, y_stds, groups, descriptor_cols, feature_names, scaler = prepare_population_statistics(train_df, mol_df)

    print("\nTraining final population models on all available data...")
    
    # Train final models on all data (not just cross-validation folds)
    final_mean_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_mean_model.fit(X, y_means)
    
    final_std_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_std_model.fit(X, y_stds)
    
    # Save the trained models, scaler, and feature names
    model_data_to_save = {
        'mean_model': final_mean_model,
        'std_model': final_std_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'descriptor_cols': descriptor_cols # Also save descriptor columns for later use
    }
    joblib.dump(model_data_to_save, POPULATION_MODELS_OUTPUT_PATH)
    print(f"Final population models, scaler, and feature names saved to {POPULATION_MODELS_OUTPUT_PATH}")

if __name__ == "__main__":
    main() 