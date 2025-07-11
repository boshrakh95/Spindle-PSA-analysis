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

def drop_nan_rows(df, column_names):
    """Drop rows where any of the specified columns has NaN values."""
    return df.dropna(subset=column_names)

def plot_combined_violin(data, variable_name, channel, path_save):
    """Plot combined violin plots for N2 and N3 stages."""
    plt.figure(figsize=(12, 6))

    # Define a custom color palette for each detector
    palette = {
        'A7': '#1f77b4',       # Blue
        'Martin': '#ff7f0e',   # Orange
        'Sumo': '#2ca02c'      # Green
    }

    # Melt data to long format for easier plotting
    melted_data = pd.melt(data, id_vars=['Detector'], value_vars=[
        f'N2_{variable_name}',
        f'N3_{variable_name}'
    ], var_name='Stage', value_name='Value')

    sns.violinplot(data=melted_data, x='Stage', y='Value', hue='Detector',
                   split=False, inner='quartile', palette=palette, cut=0)
    plt.title(f'Distribution of {variable_name} across Detectors and Stages (Channel: {channel})')
    plt.xlabel('Stage')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create directory if it doesn't exist
    plot_dir = os.path.join(path_save, 'combined_violin_plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot as PDF
    plot_path = os.path.join(plot_dir, f'combined_violin_{variable_name}_{channel}.pdf')
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"Combined violin plot saved to {plot_path}")

    plt.close()  # Close the plot to free up memory

def prepare_data(column_base_name, a7_common, martin_common, sumo_common):
    """Prepare the data for combined violin plots."""
    # Rename columns and add 'Detector' information for each dataframe
    a7_common = a7_common.copy()
    martin_common = martin_common.copy()
    sumo_common = sumo_common.copy()

    a7_common['Detector'] = 'A7'
    martin_common['Detector'] = 'Martin'
    sumo_common['Detector'] = 'Sumo'

    # Create a combined DataFrame
    combined_df = pd.concat([
        a7_common[['Detector', f'total_N2_{column_base_name}', f'total_N3_{column_base_name}']],
        martin_common[['Detector', f'total_N2_{column_base_name}', f'total_N3_{column_base_name}']],
        sumo_common[['Detector', f'total_N2_{column_base_name}', f'total_N3_{column_base_name}']]
    ], ignore_index=True)

    # Rename columns for clarity in plotting
    combined_df.rename(columns={
        f'total_N2_{column_base_name}': f'N2_{column_base_name}',
        f'total_N3_{column_base_name}': f'N3_{column_base_name}'
    }, inplace=True)

    return combined_df

def main(column_base_name, channel, path_data, path_save):
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

    # Prepare data for plotting
    combined_df = prepare_data(column_base_name, a7_common, martin_common, sumo_common)

    # Plot the combined violin plot
    plot_combined_violin(combined_df, column_base_name, channel, path_save)

if __name__ == "__main__":
    column_base_names = ['density', 'spindle_sec', 'amp_rms_uV']  # base names for columns
    path_data = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/cohort_reports'
    path_save = '/Users/boshra/Desktop/CEAMS internship/data/validation_cohort/EDF_converted/correlation_analysis/compare_detectors'

    # Process for C3 channel
    channel = 'C3'
    for col_base in column_base_names:
        print(f"Processing column base: {col_base} for channel C3")
        main(col_base, channel, path_data, path_save)
