import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_violin(data, column_name, channel, path_save):
    """Plot violin plots comparing Sumo, A7, and Martin for the variable of interest and save the plot as PDF."""
    plt.figure(figsize=(10, 6))

    # Define a custom color palette for each detector
    palette = {
        'A7': '#1f77b4',       # Blue
        'Martin': '#ff7f0e',   # Orange
        'Sumo': '#2ca02c'      # Green
    }

    sns.violinplot(data=data, x='Detector', y=column_name, inner='quartile', cut=0, palette=palette)
    plt.title(f'Distribution of {column_name} across Detectors (Channel: {channel})')
    plt.xlabel('Detector')
    plt.ylabel(column_name)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create directory if it doesn't exist
    plot_dir = os.path.join(path_save, 'violin_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot as PDF
    plot_path = os.path.join(plot_dir, f'violin_{column_name}_{channel}.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Violin plot saved to {plot_path}")

    plt.close()  # Close the plot to free up memory


def main(column_name, channel, path_data, path_save):
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
    common_filenames = set(a7_filtered['filename']).intersection(
        set(martin_filtered['filename'])).intersection(
        set(sumo_filtered['filename']))

    if not common_filenames:
        raise ValueError("No common subjects found across all detectors for the specified channel.")

    a7_common = a7_filtered[a7_filtered['filename'].isin(common_filenames)]
    martin_common = martin_filtered[martin_filtered['filename'].isin(common_filenames)]
    sumo_common = sumo_filtered[sumo_filtered['filename'].isin(common_filenames)]

    # Drop rows with NaN in the column of interest
    a7_common = drop_nan_rows(a7_common, column_name)
    martin_common = drop_nan_rows(martin_common, column_name)
    sumo_common = drop_nan_rows(sumo_common, column_name)

    # Prepare data for violin plot
    # Add detector names to each dataframe
    a7_common['Detector'] = 'A7'
    martin_common['Detector'] = 'Martin'
    sumo_common['Detector'] = 'Sumo'

    # Combine dataframes
    combined_df = pd.concat([a7_common[['Detector', column_name]],
                            martin_common[['Detector', column_name]],
                            sumo_common[['Detector', column_name]]], ignore_index=True)

    # Plot violin plot
    plot_violin(combined_df, column_name, channel, path_save)

if __name__ == "__main__":
    column_names = ['total_N2_density'  , 'total_N3_density', 'total_density', 'total_N2_spindle_sec',
                   'total_N3_spindle_sec', 'total_spindle_sec', 'total_N2_amp_rms_uV',
                   'total_N3_amp_rms_uV', 'total_amp_rms_uV']
    path_data = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/cohort_reports'
    path_save = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/correlation_analysis/compare_detectors'

    # # Process for C4 channel
    # channel = 'C4'
    # for col in column_names:
    #     print(f"Processing column: {col} for channel C4")
    #     main(col, channel, path)

    # Process for C3 channel
    channel = 'C3'
    for col in column_names:
        print(f"Processing column: {col} for channel C3")
        main(col, channel, path_data, path_save)
