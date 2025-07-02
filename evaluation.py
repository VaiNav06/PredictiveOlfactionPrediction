from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from typing import Dict, List, Tuple, Any, cast, Union
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack, issparse

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate RMSE, R^2, MAE, and Pearson Correlation."""
    # Drop NaN values for robust calculation
    combined = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    y_true_clean = combined['y_true']
    y_pred_clean = combined['y_pred']

    if len(y_true_clean) < 2:
        # For R^2 and Correlation, need at least 2 samples. RMSE/MAE can be calculated with 1.
        rmse_val = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))) if len(y_true_clean) > 0 else float('nan')
        mae_val = float(mean_absolute_error(y_true_clean, y_pred_clean)) if len(y_true_clean) > 0 else float('nan')
        return {'RMSE': rmse_val, 'R²': float('nan'), 'MAE': mae_val, 'Correlation': float('nan')}

    rmse_val = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
    r2_val = float(r2_score(y_true_clean, y_pred_clean))
    mae_val = float(mean_absolute_error(y_true_clean, y_pred_clean))
    correlation_val, _ = pearsonr(y_true_clean, y_pred_clean)
    return {'RMSE': rmse_val, 'R²': r2_val, 'MAE': mae_val, 'Correlation': float(correlation_val)}

# Constants
TEST_DATA_PATH = "Testset.txt"  # For individual predictions
LBS2_DATA_PATH = "LBs2.txt"    # For population predictions
DILUTION_DATA_PATH = "dilution_testset.txt"
MOLECULAR_DESCRIPTORS_PATH = "MolecularDescriptorData.txt"
POPULATION_MODEL_PATH = "new_population_models.joblib" # UPDATED: Path to the new population model
RF_MULTI_OUTPUT_MODEL_PATH = "rf_multioutput_pipeline.joblib" # New: Path to RF multi-output model
RESULTS_PATH = "evaluation_results.pkl" # Path to save evaluation results
INDIVIDUAL_PREDICTIONS_CSV_PATH = "individual_predictions.csv" # New: Path to save individual predictions
POPULATION_PREDICTIONS_CSV_PATH = "population_predictions.csv" # New: Path to save population predictions

def clean_string(s: str | Any) -> str:
    """Clean string by removing special characters and extra whitespace."""
    # Convert to string if not already
    s = str(s)
    # Remove quotes and whitespace
    s = s.replace('"', '').replace("'", "").strip()
    return s

def standardize_dilution(dilution: str | Any) -> str:
    """Standardize dilution format to match training data."""
    # Convert to string if not already
    dilution = str(dilution)
    # Clean string (remove quotes and whitespace)
    dilution = clean_string(dilution)
    # Remove any remaining spaces
    dilution = dilution.replace(" ", "")
    # Add 1/ prefix if missing
    if not dilution.startswith("1/"):
        dilution = "1/" + dilution
    return dilution

def load_dilution_data() -> pd.DataFrame:
    """Load and clean dilution data."""
    print("\nLoading dilution data...")
    df = pd.read_csv(DILUTION_DATA_PATH, sep='\t')
    df.columns = df.columns.str.strip()
    df['dilution'] = df['dilution'].apply(standardize_dilution)
    # Ensure oID is string and rename to Compound Identifier for consistency
    df['oID'] = df['oID'].astype(str)
    df = df.rename(columns={'oID': 'Compound Identifier'})
    df['Compound Identifier'] = df['Compound Identifier'].astype(str) # Ensure string type
    return df

def load_individual_test_data() -> pd.DataFrame:
    """Load and prepare individual test data from Testset.txt."""
    print("\nLoading individual test data...")
    
    # Load test data
    df_test = pd.read_csv(TEST_DATA_PATH, sep='\t')
    df_test.columns = df_test.columns.str.strip()
    
    # Clean up data
    df_test['descriptor'] = df_test['descriptor'].str.strip()
    df_test['value'] = pd.to_numeric(df_test['value'], errors='coerce')
    
    # Handle CID and rename columns
    df_test['CID'] = df_test['c#oID'].fillna('').astype(str)
    df_test = df_test[['CID', 'individual', 'descriptor', 'value']]
    df_test = df_test.rename(columns={'individual': 'subject #'})
    
    # Load and merge dilution data
    df_dilution = load_dilution_data()
    df_test = pd.merge(
        df_test,
        df_dilution[['Compound Identifier', 'dilution']],
        left_on='CID',
        right_on='Compound Identifier',
        how='left'
    )
    df_test = df_test.drop('Compound Identifier', axis=1)
    df_test = df_test.rename(columns={'dilution': 'Dilution'})
    
    # Convert CID to match molecular descriptors format and ensure it's string
    df_test['Compound Identifier'] = pd.to_numeric(df_test['CID'].str.replace('c#', ''), errors='coerce').astype(str)
    
    return df_test

def load_population_test_data() -> pd.DataFrame:
    """Load population test data from LBs2.txt and prepare it."""
    print("\nLoading population test data...")
    # LBs2.txt is tab-separated and has a header. Columns need cleaning and renaming.
    df_lbs2 = pd.read_csv(LBS2_DATA_PATH, sep='\t')
    
    # Clean up column names: remove # and strip whitespace
    df_lbs2.columns = [col.strip('# ') for col in df_lbs2.columns]
    
    # Ensure consistent column naming: rename 'oID' to 'Compound Identifier' if it exists
    if 'oID' in df_lbs2.columns:
        df_lbs2 = df_lbs2.rename(columns={'oID': 'Compound Identifier'})
    
    # Clean descriptor column
    df_lbs2['descriptor'] = df_lbs2['descriptor'].astype(str).str.strip()
    
    # Ensure value and sigma columns are numeric
    df_lbs2['value'] = pd.to_numeric(df_lbs2['value'], errors='coerce')
    # For sigma, handle potential percentage signs first, then convert to numeric
    df_lbs2['sigma'] = df_lbs2['sigma'].astype(str).fillna('') # Convert to string first, then fillna
    df_lbs2['sigma'] = pd.to_numeric(df_lbs2['sigma'].str.replace('%', '', regex=False), errors='coerce')

    # Ensure Compound Identifier is string for consistent merging
    df_lbs2['Compound Identifier'] = df_lbs2['Compound Identifier'].astype(str)
    
    print(f"LBs2.txt loaded. Columns: {df_lbs2.columns.tolist()}")
    print(f"LBs2.txt head:\n{df_lbs2.head()}")

    # Load and merge dilution data
    df_dilution = load_dilution_data()
    print(f"df_lbs2['Compound Identifier'] dtype: {df_lbs2['Compound Identifier'].dtype}") # Debug print
    print(f"df_dilution['Compound Identifier'] dtype: {df_dilution['Compound Identifier'].dtype}") # Debug print

    df_lbs2 = pd.merge(
        df_lbs2,
        df_dilution[['Compound Identifier', 'dilution']], # Use Compound Identifier from dilution data
        on='Compound Identifier', # Merge on consistent column name
        how='left'
    )
    df_lbs2 = df_lbs2.rename(columns={'dilution': 'Dilution'})
    
    return df_lbs2

def load_molecular_descriptors() -> pd.DataFrame:
    """Load molecular descriptors data."""
    print("\nLoading molecular descriptors...")
    df = pd.read_csv(MOLECULAR_DESCRIPTORS_PATH, sep="\t", encoding='utf-8')
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'CID': 'Compound Identifier'})
    # Ensure Compound Identifier is string for consistent merging
    df['Compound Identifier'] = df['Compound Identifier'].astype(str)
    return df

def prepare_features_for_rf_multioutput_model(df: pd.DataFrame, mol_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features (raw) for the RF MultiOutput model, including subject ID and dilution."""
    print("\nPreparing features for RF MultiOutput model...")

    # Merge with molecular descriptors
    df_merged = pd.merge(df, mol_df, on='Compound Identifier', how='inner')
    
    # Select relevant columns for the model input (subject #, Dilution, and all molecular descriptors)
    molecular_descriptor_cols = [col for col in mol_df.columns if col != 'Compound Identifier']
    input_cols = ['subject #', 'Dilution'] + molecular_descriptor_cols
    
    # Ensure all required columns are present before returning
    # The MultiOutputRegressor's pipeline will handle imputation, encoding, and scaling
    X_input = cast(pd.DataFrame, df_merged[input_cols].copy())
    
    return X_input

def prepare_population_features(df: pd.DataFrame, mol_df: pd.DataFrame, scaler: StandardScaler, expected_feature_names: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Prepare features for population predictions, using components from the trained population model data."""
    print("\nPreparing population features (using new method)...")

    # Merge with molecular descriptors
    df_merged = pd.merge(df, mol_df, on='Compound Identifier', how='inner')

    # One-hot encode dilution (manually, as preprocessor.joblib is not used for this pipeline)
    # Ensure consistent columns with what the model was trained on.
    dilution_dummies = pd.get_dummies(df_merged['Dilution'], prefix='dilution').astype(float)
    
    # Identify which dilution columns were expected by the model during training
    expected_dilution_cols = [col for col in expected_feature_names if col.startswith('dilution_')]
    
    # Add missing dilution columns (with 0s) if new dilutions appear in test set, and reorder
    for col in expected_dilution_cols:
        if col not in dilution_dummies.columns:
            dilution_dummies[col] = 0.0
    dilution_dummies = dilution_dummies[expected_dilution_cols]

    # Prepare molecular descriptors
    molecular_descriptor_cols = [col for col in mol_df.columns if col != 'Compound Identifier']
    # Ensure all expected molecular descriptor columns are present, fill missing with 0 (imputer handles during training)
    molecular_data = df_merged[molecular_descriptor_cols].copy()
    for col in molecular_descriptor_cols:
        if col not in molecular_data.columns:
            molecular_data[col] = 0.0 # Or np.nan if imputer expects NaN, but 0 is common for missing molecular values
    
    # Combine the prepared categorical and molecular descriptor data
    # The order of concatenation must match the order in expected_feature_names
    # Ensure all parts are numpy arrays of float before hstack
    combined_features_df = pd.concat([dilution_dummies, molecular_data], axis=1)
    
    # Align columns to the order expected by the model (feature_names from new_population_models.joblib)
    # This is crucial to match the input features with the trained model's expectation
    X_combined = combined_features_df[expected_feature_names].values.astype(float)
    
    # Apply the scaler from the loaded model data
    X_processed = cast(np.ndarray, scaler.transform(X_combined))

    print(f"Shape of processed population features: {X_processed.shape}")
    # Return the processed features along with the corresponding identifiers for merging
    print(f"df_merged columns before returning identifiers: {df_merged.columns.tolist()[:10]}...") # Debug print
    return X_processed, cast(pd.DataFrame, df_merged[['Compound Identifier', 'Dilution']].copy())

def main():
    """Main function to evaluate both individual and population predictions."""
    # Load common data
    mol_df = load_molecular_descriptors()
    
    # Load individual prediction model (RF MultiOutput Pipeline)
    print("\nLoading RF MultiOutput Pipeline model...")
    rf_multioutput_pipeline = joblib.load(RF_MULTI_OUTPUT_MODEL_PATH)

    # Load the population model components (mean_model, std_model, scaler, feature_names)
    print("\nLoading Population Model components...")
    population_model_data = joblib.load(POPULATION_MODEL_PATH)
    mean_model = population_model_data['mean_model']
    std_model = population_model_data['std_model']
    # scaler and feature_names are now passed to prepare_population_features directly

    print("\nEvaluating individual predictions (using RF MultiOutput Model)...")
    individual_df_raw = load_individual_test_data()

    # Get unique combinations of Compound Identifier, subject #, and Dilution for predictions
    unique_individual_samples_for_prediction = individual_df_raw.drop_duplicates(subset=['Compound Identifier', 'subject #', 'Dilution']).copy()

    # Prepare features for prediction. X_individual_rf_input will have one row per unique sample.
    X_individual_rf_input = prepare_features_for_rf_multioutput_model(unique_individual_samples_for_prediction, mol_df)
    
    # Make predictions for all descriptors at once using the RF pipeline
    # all_individual_predictions_rf will have shape (num_unique_samples, num_descriptors)
    all_individual_predictions_rf = rf_multioutput_pipeline.predict(X_individual_rf_input)
    
    # Create a DataFrame for individual predictions (mean for each unique sample-descriptor)
    # The index should match the unique samples that were used for prediction
    predicted_values_for_individual_df = pd.DataFrame(
        all_individual_predictions_rf,
        columns=cast(List[str], [f'{desc}_predicted' for desc in population_model_data['descriptor_cols']]),
        index=unique_individual_samples_for_prediction.index # Use unique_individual_samples_for_prediction index
    )

    # Merge the unique samples with their predictions
    individual_predictions_to_save = unique_individual_samples_for_prediction.copy()
    for col in predicted_values_for_individual_df.columns:
        individual_predictions_to_save[col] = predicted_values_for_individual_df[col]

    # Save individual predictions to CSV
    individual_predictions_to_save.to_csv(INDIVIDUAL_PREDICTIONS_CSV_PATH, index=False)
    print(f"Individual predictions saved to {INDIVIDUAL_PREDICTIONS_CSV_PATH}")

    # Initialize evaluation results dictionary
    evaluation_results: Dict[str, Dict[str, Dict[str, float]]] = {
        'individual_predictions': {},
        'population_mean_predictions': {},
        'population_std_predictions': {}
    }

    # Evaluate individual predictions (per descriptor)
    print("\nCalculating metrics for individual predictions...")
    unique_descriptors = population_model_data['descriptor_cols']
    for desc in unique_descriptors:
        # Filter individual_df_raw for the current descriptor (true values)
        y_true_individual = individual_df_raw[individual_df_raw['descriptor'] == desc]['value']
        
        # Get the corresponding predicted values. Note: individual_predictions_to_save has one row per unique sample.
        # We need to map these back to the original individual_df_raw structure for true values.
        # This requires careful alignment by 'Compound Identifier', 'subject #', and 'Dilution'.
        # A simpler approach for evaluation here is to assume `all_individual_predictions_rf` is ordered
        # consistently with `unique_individual_samples_for_prediction` and `individual_df_raw` for this descriptor.
        # However, for robust evaluation, it's better to merge true and predicted values based on identifiers.

        # For individual predictions, let's re-merge with true values for robust metric calculation
        # Create a temporary dataframe for individual predictions in long format for easy merging
        # This assumes individual_predictions_to_save has 'Compound Identifier', 'subject #', 'Dilution', and descriptor_predicted columns.
        temp_individual_predictions_long = individual_predictions_to_save.melt(
            id_vars=['Compound Identifier', 'subject #', 'Dilution'],
            value_vars=[f'{d}_predicted' for d in unique_descriptors],
            var_name='descriptor_predicted', value_name='predicted_value'
        )
        temp_individual_predictions_long['descriptor'] = temp_individual_predictions_long['descriptor_predicted'].str.replace('_predicted', '')
        
        # Merge true values from individual_df_raw with predicted values
        merged_individual_eval = pd.merge(
            cast(pd.DataFrame, individual_df_raw[individual_df_raw['descriptor'] == desc][['Compound Identifier', 'subject #', 'Dilution', 'value']]),
            cast(pd.DataFrame, temp_individual_predictions_long[temp_individual_predictions_long['descriptor'] == desc][['Compound Identifier', 'subject #', 'Dilution', 'predicted_value']]),
            on=['Compound Identifier', 'subject #', 'Dilution'],
            how='inner'
        )

        if not merged_individual_eval.empty:
            metrics = calculate_metrics(
                y_true=cast(pd.Series, merged_individual_eval['value']),
                y_pred=cast(pd.Series, merged_individual_eval['predicted_value'])
            )
            evaluation_results['individual_predictions'][desc] = metrics
        else:
            print(f"Warning: No matching data found for individual prediction evaluation of descriptor: {desc}")
            evaluation_results['individual_predictions'][desc] = {'RMSE': float('nan'), 'R²': float('nan'), 'MAE': float('nan'), 'Correlation': float('nan')}

    # Evaluate population predictions (using POPULATION_MODEL_PATH)
    print("\nEvaluating population predictions (using Population Model)...")
    population_df_raw = load_population_test_data()
    X_population, population_base_identifiers = prepare_population_features(
        population_df_raw, 
        mol_df, 
        population_model_data['scaler'], 
        population_model_data['feature_names']
    )
    
    # Make predictions for mean and std
    all_mean_predictions_pop = mean_model.predict(X_population)
    all_std_predictions_pop = std_model.predict(X_population)

    print(f"Shape of all_mean_predictions_pop: {all_mean_predictions_pop.shape}")
    print(f"Shape of all_std_predictions_pop: {all_std_predictions_pop.shape}")

    # Convert wide predictions to long format for saving and evaluation
    # Create a DataFrame with identifiers and wide predictions
    wide_predictions_df = population_base_identifiers.copy()
    
    # Add predicted mean columns
    for i, desc in enumerate(population_model_data['descriptor_cols']):
        wide_predictions_df[f'{desc}_mean_predicted'] = all_mean_predictions_pop[:, i]
    
    # Add predicted std columns
    for i, desc in enumerate(population_model_data['descriptor_cols']):
        wide_predictions_df[f'{desc}_std_predicted'] = all_std_predictions_pop[:, i]

    # Melt the wide predictions DataFrame to long format
    # This creates one row per descriptor prediction, which is used for saving and evaluation.
    id_vars = ['Compound Identifier', 'Dilution']
    value_vars_mean = [f'{desc}_mean_predicted' for desc in population_model_data['descriptor_cols']]
    value_vars_std = [f'{desc}_std_predicted' for desc in population_model_data['descriptor_cols']]
    
    melted_mean_predictions = pd.melt(wide_predictions_df, id_vars=id_vars, value_vars=value_vars_mean,
                                      var_name='descriptor_mean', value_name='predicted_mean')
    melted_std_predictions = pd.melt(wide_predictions_df, id_vars=id_vars, value_vars=value_vars_std,
                                     var_name='descriptor_std', value_name='predicted_std')

    # Clean up descriptor names after melting
    melted_mean_predictions['descriptor'] = melted_mean_predictions['descriptor_mean'].str.replace('_mean_predicted', '')
    melted_std_predictions['descriptor'] = melted_std_predictions['descriptor_std'].str.replace('_std_predicted', '')

    # Merge mean and std predictions back together into a single long format DataFrame
    population_predictions_df = pd.merge(
        cast(pd.DataFrame, melted_mean_predictions[['Compound Identifier', 'Dilution', 'descriptor', 'predicted_mean']]),
        cast(pd.DataFrame, melted_std_predictions[['Compound Identifier', 'Dilution', 'descriptor', 'predicted_std']]),
        on=['Compound Identifier', 'Dilution', 'descriptor'],
        how='inner'
    )

    # Save population predictions to CSV
    population_predictions_df.to_csv(POPULATION_PREDICTIONS_CSV_PATH, index=False)
    print(f"Population predictions saved to {POPULATION_PREDICTIONS_CSV_PATH}")

    # Now, merge population_predictions_df with original population_df_raw for evaluation
    # Ensure 'Compound Identifier' and 'Dilution' are consistent types before merging
    population_predictions_df['Compound Identifier'] = population_predictions_df['Compound Identifier'].astype(str)
    population_predictions_df['Dilution'] = population_predictions_df['Dilution'].astype(str)
    population_df_raw['Compound Identifier'] = population_df_raw['Compound Identifier'].astype(str)
    population_df_raw['Dilution'] = population_df_raw['Dilution'].astype(str)

    merged_population_eval = pd.merge(
        cast(pd.DataFrame, population_df_raw[['Compound Identifier', 'Dilution', 'descriptor', 'value', 'sigma']]),
        cast(pd.DataFrame, population_predictions_df[['Compound Identifier', 'Dilution', 'descriptor', 'predicted_mean', 'predicted_std']]),
        on=['Compound Identifier', 'Dilution', 'descriptor'],
        how='inner'
    )

    print(f"Shape of merged_population_eval for metrics: {merged_population_eval.shape}")
    
    # Calculate and store metrics for population mean and std predictions per descriptor
    for desc in unique_descriptors:
        filtered_eval_data = merged_population_eval[merged_population_eval['descriptor'] == desc]
        
        # Evaluate mean predictions
        if not filtered_eval_data.empty:
            metrics_mean = calculate_metrics(
                y_true=cast(pd.Series, filtered_eval_data['value']),
                y_pred=cast(pd.Series, filtered_eval_data['predicted_mean'])
            )
            evaluation_results['population_mean_predictions'][desc] = metrics_mean
        else:
            print(f"Warning: No matching data found for population mean prediction evaluation of descriptor: {desc}")
            evaluation_results['population_mean_predictions'][desc] = {'RMSE': float('nan'), 'R²': float('nan'), 'MAE': float('nan'), 'Correlation': float('nan')}

        # Evaluate std predictions (against sigma if available/appropriate)
        if not filtered_eval_data.empty and 'sigma' in filtered_eval_data.columns:
            metrics_std = calculate_metrics(
                y_true=cast(pd.Series, filtered_eval_data['sigma']),
                y_pred=cast(pd.Series, filtered_eval_data['predicted_std'])
            )
            evaluation_results['population_std_predictions'][desc] = metrics_std
        else:
            print(f"Warning: No matching data found for population std prediction evaluation of descriptor: {desc} or sigma not available.")
            evaluation_results['population_std_predictions'][desc] = {'RMSE': float('nan'), 'R²': float('nan'), 'MAE': float('nan'), 'Correlation': float('nan')}

    # Save the full evaluation results to a .pkl file
    joblib.dump(evaluation_results, RESULTS_PATH)
    print(f"Evaluation results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main() 