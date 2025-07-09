import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np
import os

def read_and_sort(file_path):
    """Read a TSV file and sort by filename."""
    df = pd.read_csv(file_path, sep='\t')
    return df.sort_values('filename')

def filter_by_channel(df, channel):
    """Filter DataFrame by channel label."""
    return df[df['chan_label'].str.contains(channel, case=False, regex=False)]

def drop_nan_rows(df, column_name):
    """Drop rows where the specified column has NaN values."""
    return df.dropna(subset=[column_name])

def plot_correlation(ax, x, y, xlabel, ylabel, feature_name, show_stats=True):
    """Plot correlation with optional regression line and stats."""
    if show_stats:
        # Remove NaN pairs before plotting and calculating stats
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        sns.regplot(x=x_clean, y=y_clean, ax=ax)
        if len(x_clean) > 1:  # Need at least 2 points for regression
            slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
            r_squared = r_value**2
            std_dev = y_clean.std()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{xlabel} vs {ylabel} ({feature_name})')
            ax.text(0.05, 0.95, f'R² = {r_squared:.2f}\nStd = {std_dev:.2f}\nN = {len(x_clean)}',
                   transform=ax.transAxes, verticalalignment='top')
        else:
            ax.scatter(x_clean, y_clean)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{xlabel} vs {ylabel} ({feature_name})')
            ax.text(0.05, 0.95, 'Insufficient data for regression',
                   transform=ax.transAxes, verticalalignment='top')
    else:
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        ax.scatter(x_clean, y_clean)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel} ({feature_name})')

def main(column_name, channel, path, show_stats=True):
    # Construct file paths for the two SUMO reports
    sumo_artifact_removed_path = os.path.join(path, 'SUMO/artifact_removed/sumo_val_artifact_removed.tsv')
    sumo_artifact_not_removed_path = os.path.join(path, 'SUMO/artifact_not_removed/sumo_val_artifact_not_removed.tsv')

    # Check if files exist
    for file_path in [sumo_artifact_removed_path, sumo_artifact_not_removed_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read and sort files by filename
    sumo_removed_df = read_and_sort(sumo_artifact_removed_path)
    sumo_not_removed_df = read_and_sort(sumo_artifact_not_removed_path)

    # Check if column exists in both DataFrames
    for name, df in [('SUMO (Artifacts Removed)', sumo_removed_df), ('SUMO (Artifacts Not Removed)', sumo_not_removed_df)]:
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the {name} DataFrame.")

    # Filter by channel
    sumo_removed_filtered = filter_by_channel(sumo_removed_df, channel)
    sumo_not_removed_filtered = filter_by_channel(sumo_not_removed_df, channel)

    # Find common subjects using filename
    common_filenames = set(sumo_removed_filtered['filename']).intersection(set(sumo_not_removed_filtered['filename']))
    if not common_filenames:
        raise ValueError("No common subjects found across both SUMO reports for the specified channel.")

    sumo_removed_common = sumo_removed_filtered[sumo_removed_filtered['filename'].isin(common_filenames)]
    sumo_not_removed_common = sumo_not_removed_filtered[sumo_not_removed_filtered['filename'].isin(common_filenames)]

    # Drop rows with NaN in the column of interest
    sumo_removed_common = drop_nan_rows(sumo_removed_common, column_name)
    sumo_not_removed_common = drop_nan_rows(sumo_not_removed_common, column_name)

    # Get column data
    sumo_removed_data = sumo_removed_common[column_name]
    sumo_not_removed_data = sumo_not_removed_common[column_name]

    # Check if data is empty
    if sumo_removed_data.empty or sumo_not_removed_data.empty:
        raise ValueError("One or more datasets are empty after filtering.")

    # Find indices where both datasets have valid values
    valid_mask = ~pd.isna(sumo_removed_data) & ~pd.isna(sumo_not_removed_data)
    sumo_removed_clean = sumo_removed_data[valid_mask]
    sumo_not_removed_clean = sumo_not_removed_data[valid_mask]

    # Check if we have any valid pairs left
    if len(sumo_removed_clean) == 0:
        print(f"No valid data points for {column_name} with channel {channel}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_correlation(ax, sumo_removed_clean, sumo_not_removed_clean,
                    'SUMO (Artifacts Removed)', 'SUMO (Artifacts Not Removed)',
                    column_name, show_stats)

    plt.tight_layout()

    plot_dir = os.path.join(path, 'sumo_robustness', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'sumo_comparison_{column_name}_{channel}.pdf')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    plt.show()

if __name__ == "__main__":
    column_names = ['total_N2_density', 'total_N3_density', 'total_density', 'total_N2_spindle_sec', 'total_N3_spindle_sec',
                   'total_spindle_sec', 'total_N2_amp_rms_uV', 'total_N3_amp_rms_uV', 'total_amp_rms_uV']

    path = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/cohort_reports'
    show_stats = True  # Toggle to show/hide regression line and stats


    channel = 'C3'
    # Process each column for channel C3
    for col in column_names:
        print(f"Processing column: {col} for channel C3")
        main(col, channel, path, show_stats)

    channel = 'C4'
    for col in column_names:
        print(f"Processing column: {col} for channel C4")
        main(col, channel, path, show_stats)
