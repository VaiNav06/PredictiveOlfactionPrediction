import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Constants ---
EVALUATION_RESULTS_CSV = "evaluation_results.csv"
PLOTS_DIR = "plots/"

def plot_comparison_bar_chart(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, filename: str):
    """Generates and saves a bar chart comparing a given metric for multiple prediction types."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    descriptors = sorted(df['Descriptor'].unique())

    # Define the prediction types and their colors
    prediction_types = [
        ('individual_predictions', 'Individual Predictions', 'skyblue'),
        ('population_mean_predictions', 'Population Mean Predictions', 'lightcoral'),
        ('population_std_predictions', 'Population Standard Deviation Predictions', 'lightgreen')
    ]

    bar_width = 0.2
    index = np.arange(len(descriptors))

    for i, (type_key, type_label, color) in enumerate(prediction_types):
        df_filtered = df[df['Prediction Type'] == type_key].set_index('Descriptor')
        values = [df_filtered.loc[d, metric_col] if d in df_filtered.index else np.nan for d in descriptors]
        ax.bar(index + i * bar_width, values, bar_width, label=type_label, color=color)

    ax.set_xlabel('Descriptor')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (len(prediction_types) - 1) / 2)
    ax.set_xticklabels(descriptors, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    print(f"Saved {filename} to {PLOTS_DIR}")
    plt.close() 

def main():
    print("\nGenerating comparison plots...")
    try:
        df_results = pd.read_csv(EVALUATION_RESULTS_CSV)
    except FileNotFoundError:
        print(f"Error: {EVALUATION_RESULTS_CSV} not found. Please ensure evaluation.py has been run successfully.")
        return
    
    # Generate plots for RMSE, R², MAE, and Correlation for all prediction types
    plot_comparison_bar_chart(
        df_results, 
        'RMSE', 
        'RMSE Comparison: All Prediction Types',
        'RMSE Value',
        'RMSE_all_types_comparison_bar.png'
    )
    
    plot_comparison_bar_chart(
        df_results, 
        'R²', 
        'R² Comparison: All Prediction Types',
        'R² Value',
        'R2_all_types_comparison_bar.png'
    )
    
    plot_comparison_bar_chart(
        df_results, 
        'MAE', 
        'MAE Comparison: All Prediction Types',
        'MAE Value',
        'MAE_all_types_comparison_bar.png'
    )

    plot_comparison_bar_chart(
        df_results, 
        'Correlation', 
        'Correlation Comparison: All Prediction Types',
        'Correlation Value',
        'Correlation_all_types_comparison_bar.png'
    )

if __name__ == "__main__":
    main() 