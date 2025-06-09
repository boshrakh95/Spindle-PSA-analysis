# Modified code for plotting N3 (Avoid negative values)

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# Load the TSV file
file_path = '/Users/boshra/Downloads/karine_PSA_power_nettoye_transposed.tsv'
specific_figure_path = '/Users/boshra/Desktop/CEAMS internship/data/figures_PSA/modified/'
data = pd.read_csv(file_path, sep='\t')

# Extract relevant columns for N2 and N3 stages
n3_columns = [col for col in data.columns if 'N3_act_abs_ROI_C3C4' in col]

# Filter out subjects without N3 data
data = data.dropna(subset=n3_columns, how='all')

# Extract the group information
groups = data['Groups (IH-1 controls=0)']

# Extract the absolute power data for N3
n3_data = data[n3_columns]

# Compute the total power across 1-32 Hz for N3
total_n3_power = n3_data.sum(axis=1)

# Separate data by group
control_group_data = n3_data[groups == 0]
ih_group_data = n3_data[groups == 1]

# Calculate mean and standard deviation for each frequency bin in each group
control_group_mean = control_group_data.mean(axis=0)
control_group_std = control_group_data.std(axis=0)
ih_group_mean = ih_group_data.mean(axis=0)
ih_group_std = ih_group_data.std(axis=0)

# Calculate mean - std for each frequency bin in each group
control_group_min = control_group_mean - control_group_std
ih_group_min = ih_group_mean - ih_group_std

# Find the smallest resulting value from mean - std
min_mean_std = min(control_group_min.min(), ih_group_min.min())

# Use absolute value and add a small constant
shift_value = abs(min_mean_std) + 1e-5

# Shift both groups' data by the shift value
shifted_control_group_data = control_group_data + shift_value
shifted_ih_group_data = ih_group_data + shift_value

# Define colors for the groups and their standard deviations
colors = ['#1f77b4', '#ff7f0e']
std_colors = ['#a6cee3', '#fdbf6f']  # Lighter colors for the standard deviation areas

# Function to plot and save the power spectral curves using Matplotlib with shaded areas
def plot_and_save_power_spectral_curves(shifted_control_group_data, shifted_ih_group_data, stage, specific_figure_path, log_scale=False):
    plt.figure(figsize=(18, 6))

    # Plot absolute power
    plt.subplot(1, 2, 1)
    for idx, (group_data, group_label) in enumerate([(shifted_control_group_data, 'Control'), (shifted_ih_group_data, 'IH')]):
        mean = group_data.mean(axis=0)
        std = group_data.std(axis=0)
        plt.plot(mean, color=colors[idx], label=group_label)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color=std_colors[idx], alpha=0.5)
    plt.title(f'Absolute Power for {stage} {"(Log Scale)" if log_scale else ""}')
    plt.xlabel('Frequency Bins (Hz)')
    plt.ylabel('Absolute Power')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.xticks(np.arange(len(n3_columns)), [col.split('_')[-2] for col in n3_columns], rotation=90)

    # Plot relative power
    plt.subplot(1, 2, 2)
    total_power_control = shifted_control_group_data.sum(axis=1)
    relative_control_group_data = shifted_control_group_data.div(total_power_control, axis=0)
    total_power_ih = shifted_ih_group_data.sum(axis=1)
    relative_ih_group_data = shifted_ih_group_data.div(total_power_ih, axis=0)

    for idx, (group_data, group_label) in enumerate([(relative_control_group_data, 'Control'), (relative_ih_group_data, 'IH')]):
        mean = group_data.mean(axis=0)
        std = group_data.std(axis=0)
        plt.plot(mean, color=colors[idx], label=group_label)
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color=std_colors[idx], alpha=0.5)
    plt.title(f'Relative Power for {stage} {"(Log Scale)" if log_scale else ""}')
    plt.xlabel('Frequency Bins (Hz)')
    plt.ylabel('Relative Power')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.xticks(np.arange(len(n3_columns)), [col.split('_')[-2] for col in n3_columns], rotation=90)

    plt.tight_layout()

    # Save the figure as a PDF file
    scale_type = "log" if log_scale else "linear"
    filename_fig = f'power_spectral_curves_{stage}_{scale_type}_shaded'
    plt.savefig(f"{specific_figure_path}{filename_fig}2.pdf")
    plt.close()


# Plot and save the power spectral curves for N3 with shaded areas
plot_and_save_power_spectral_curves(shifted_control_group_data, shifted_ih_group_data, 'N3', specific_figure_path, log_scale=False)
plot_and_save_power_spectral_curves(shifted_control_group_data, shifted_ih_group_data, 'N3', specific_figure_path, log_scale=True)
