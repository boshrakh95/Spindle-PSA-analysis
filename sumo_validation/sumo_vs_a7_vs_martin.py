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
    # Remove NaN pairs before plotting and calculating stats
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    if show_stats:
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
        ax.scatter(x_clean, y_clean)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel} ({feature_name})')

    # Set equal axis ranges
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add identity line (y = x)
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

def main(column_name, channel, path_data, path_save, show_stats=True):
    # Construct file paths
    a7_path = os.path.join(path_data, 'A7/artifact_removed/a7_val_artifact_removed.tsv')
    martin_path = os.path.join(path_data, 'Martin/artifact_removed/martin_val_artifact_removed.tsv')
    sumo_path = os.path.join(path_data, 'SUMO/artifact_removed/sumo_val_artifact_removed.tsv')

    # Check if files exist
    for file_path in [a7_path, martin_path, sumo_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read and sort files by filename
    a7_df = read_and_sort(a7_path)
    martin_df = read_and_sort(martin_path)
    sumo_df = read_and_sort(sumo_path)

    # Check if column exists in all DataFrames
    for name, df in [('A7', a7_df), ('Martin', martin_df), ('Sumo', sumo_df)]:
        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the {name} DataFrame.")

    # Filter by channel
    a7_filtered = filter_by_channel(a7_df, channel)
    martin_filtered = filter_by_channel(martin_df, channel)
    sumo_filtered = filter_by_channel(sumo_df, channel)

    # Find common subjects using filename
    common_filenames = set(a7_filtered['filename']).intersection(set(martin_filtered['filename'])).intersection(set(sumo_filtered['filename']))
    if not common_filenames:
        raise ValueError("No common subjects found across all detectors for the specified channel.")

    a7_common = a7_filtered[a7_filtered['filename'].isin(common_filenames)]
    martin_common = martin_filtered[martin_filtered['filename'].isin(common_filenames)]
    sumo_common = sumo_filtered[sumo_filtered['filename'].isin(common_filenames)]

    # Drop rows with NaN in the column of interest
    a7_common = drop_nan_rows(a7_common, column_name)
    martin_common = drop_nan_rows(martin_common, column_name)
    sumo_common = drop_nan_rows(sumo_common, column_name)

    # Get column data
    a7_data = a7_common[column_name]
    martin_data = martin_common[column_name]
    sumo_data = sumo_common[column_name]

    # Check if data is empty
    if a7_data.empty or martin_data.empty or sumo_data.empty:
        raise ValueError("One or more datasets are empty after filtering.")

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Filter out NaN pairs for each comparison
    # A7 vs Martin
    valid_mask = ~np.isnan(a7_data) & ~np.isnan(martin_data)
    a7_clean = a7_data[valid_mask]
    martin_clean = martin_data[valid_mask]

    # A7 vs Sumo
    valid_mask2 = ~np.isnan(a7_data) & ~np.isnan(sumo_data)
    a7_clean2 = a7_data[valid_mask2]
    sumo_clean = sumo_data[valid_mask2]

    # Martin vs Sumo
    valid_mask3 = ~np.isnan(martin_data) & ~np.isnan(sumo_data)
    martin_clean2 = martin_data[valid_mask3]
    sumo_clean2 = sumo_data[valid_mask3]

    # Plot correlations with filtered data
    plot_correlation(axes[0], a7_clean, martin_clean, 'A7', 'Martin', column_name, show_stats)
    plot_correlation(axes[1], a7_clean2, sumo_clean, 'A7', 'Sumo', column_name, show_stats)
    plot_correlation(axes[2], martin_clean2, sumo_clean2, 'Martin', 'Sumo', column_name, show_stats)

    plt.tight_layout()

    # Create 'plot' directory if it doesn't exist
    plot_dir = os.path.join(path_save, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot
    plot_path = os.path.join(plot_dir, f'correlation_{column_name}_{channel}.pdf')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    plt.close()

if __name__ == "__main__":
    column_names = ['total_N2_density', 'total_N3_density', 'total_density', 'total_N2_spindle_sec',
                   'total_N3_spindle_sec', 'total_spindle_sec', 'total_N2_amp_rms_uV',
                   'total_N3_amp_rms_uV', 'total_amp_rms_uV']
    path_data = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/cohort_reports'
    path_save = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/correlation_analysis/compare_detectors'
    show_stats = True

    # Process for C4 channel
    # channel = 'C4'
    # for col in column_names:
    #     print(f"Processing column: {col} for channel C4")
    #     main(col, channel, path, show_stats)

    # Process for C3 channel
    channel = 'C3'
    for col in column_names:
        print(f"Processing column: {col} for channel C3")
        main(col, channel, path_data, path_save, show_stats)


