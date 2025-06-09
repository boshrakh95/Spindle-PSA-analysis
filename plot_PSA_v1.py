import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# Load the TSV file
file_path = '/Users/boshra/Downloads/karine_PSA_power_nettoye_transposed.tsv'
data = pd.read_csv(file_path, sep='\t')

# Specify the directory where you want to save the figures
specific_figure_path = '/Users/boshra/Desktop/CEAMS internship/data/figures_PSA/modified/'

# Extract relevant columns for N2 and N3 stages
n2_columns = [col for col in data.columns if 'N2_act_abs_ROI_C3C4' in col]
n3_columns = [col for col in data.columns if 'N3_act_abs_ROI_C3C4' in col]

# Filter out subjects without N3 data
data = data.dropna(subset=n3_columns, how='all')

# Extract the group information
groups = data['Groups (IH-1 controls=0)']

# Extract the absolute power data for N2 and N3
n2_data = data[n2_columns]
n3_data = data[n3_columns]

# Compute the total power across 1-32 Hz for N2 and N3
total_n2_power = n2_data.sum(axis=1)
total_n3_power = n3_data.sum(axis=1)

# Compute the relative power for N2 and N3
relative_n2_data = n2_data.div(total_n2_power, axis=0)
relative_n3_data = n3_data.div(total_n3_power, axis=0)

# Define colors for the groups and their standard deviations
colors = ['#1f77b4', '#ff7f0e']
std_colors = ['#a6cee3', '#fdbf6f']  # Lighter colors for the standard deviation areas


# # Function to plot and save the power spectral curves using Matplotlib
# def plot_and_save_power_spectral_curves(absolute_data, relative_data, stage, specific_figure_path, log_scale=False):
#     plt.figure(figsize=(18, 6))
#
#     # Plot absolute power
#     plt.subplot(1, 2, 1)
#     for idx, group in enumerate([0, 1]):
#         group_data = absolute_data[groups == group]
#         group_label = 'Control' if group == 0 else 'IH'
#         mean = group_data.mean(axis=0)
#         std = group_data.std(axis=0)
#         plt.plot(mean, color=colors[idx], label=group_label)
#         plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color=std_colors[idx], alpha=0.3)
#     plt.title(f'Absolute Power for {stage}')
#     plt.xlabel('Frequency Bins (Hz)')
#     plt.ylabel('Absolute Power')
#     if log_scale:
#         plt.yscale('log')
#     plt.legend()
#     plt.xticks(np.arange(len(n2_columns)), [col.split('_')[-2] for col in n2_columns], rotation=90)
#
#     # Plot relative power
#     plt.subplot(1, 2, 2)
#     for idx, group in enumerate([0, 1]):
#         group_data = relative_data[groups == group]
#         group_label = 'Control' if group == 0 else 'IH'
#         mean = group_data.mean(axis=0)
#         std = group_data.std(axis=0)
#         plt.plot(mean, color=colors[idx], label=group_label)
#         plt.fill_between(np.arange(len(mean)), mean - std, mean + std, color=std_colors[idx], alpha=0.3)
#     plt.title(f'Relative Power for {stage}')
#     plt.xlabel('Frequency Bins (Hz)')
#     plt.ylabel('Relative Power')
#     if log_scale:
#         plt.yscale('log')
#     plt.legend()
#     plt.xticks(np.arange(len(n2_columns)), [col.split('_')[-2] for col in n2_columns], rotation=90)
#
#     plt.tight_layout()
#
#     # Save the figure as a PDF file
#     scale_type = "log" if log_scale else "linear"
#     filename_fig = f'power_spectral_curves_{stage}_{scale_type}'
#     plt.savefig(specific_figure_path + filename_fig + ".pdf")
#     plt.close()
#
# # Specify the directory where you want to save the figures
# specific_figure_path = '/Users/boshra/Desktop/CEAMS internship/data/figures_PSA/modified/'
#
# # Plot and save the power spectral curves for N2 and N3 with linear scale
# plot_and_save_power_spectral_curves(n2_data, relative_n2_data, 'N2', specific_figure_path, log_scale=False)
# plot_and_save_power_spectral_curves(n3_data, relative_n3_data, 'N3', specific_figure_path, log_scale=False)
#
# # Plot and save the power spectral curves for N2 and N3 with logarithmic scale
# plot_and_save_power_spectral_curves(n2_data, relative_n2_data, 'N2', specific_figure_path, log_scale=True)
# plot_and_save_power_spectral_curves(n3_data, relative_n3_data, 'N3', specific_figure_path, log_scale=True)


################

# Function to plot and save the power spectral curves using Matplotlib with shaded areas
def plot_and_save_power_spectral_curves(absolute_data, relative_data, stage, specific_figure_path, log_scale=False):
    plt.figure(figsize=(18, 6))

    # Plot absolute power
    plt.subplot(1, 2, 1)
    for idx, group in enumerate([0, 1]):
        group_data = absolute_data[groups == group]
        group_label = 'Control' if group == 0 else 'IH'
        mean = group_data.mean(axis=0)
        std = group_data.std(axis=0)
        plt.plot(mean, color=colors[idx], label=group_label)
        plt.fill_between(np.arange(len(mean)), abs(mean - std), mean + std, color=std_colors[idx], alpha=0.5)
    plt.title(f'Absolute Power for {stage} {"(Log Scale)" if log_scale else ""}')
    plt.xlabel('Frequency Bins (Hz)')
    plt.ylabel('Absolute Power')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.xticks(np.arange(len(n2_columns)), [col.split('_')[-2] for col in n2_columns], rotation=90)

    # Plot relative power
    plt.subplot(1, 2, 2)
    for idx, group in enumerate([0, 1]):
        group_data = relative_data[groups == group]
        group_label = 'Control' if group == 0 else 'IH'
        mean = group_data.mean(axis=0)
        std = group_data.std(axis=0)
        plt.plot(mean, color=colors[idx], label=group_label)
        plt.fill_between(np.arange(len(mean)), abs(mean - std), mean + std, color=std_colors[idx], alpha=0.5)
    plt.title(f'Relative Power for {stage} {"(Log Scale)" if log_scale else ""}')
    plt.xlabel('Frequency Bins (Hz)')
    plt.ylabel('Relative Power')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.xticks(np.arange(len(n2_columns)), [col.split('_')[-2] for col in n2_columns], rotation=90)

    plt.tight_layout()

    # Save the figure as a PDF file
    scale_type = "log" if log_scale else "linear"
    filename_fig = f'power_spectral_curves_{stage}_{scale_type}_shaded'
    plt.savefig(f"{specific_figure_path}{filename_fig}.pdf")
    plt.close()


# Plot and save the power spectral curves for N2 and N3 with shaded areas
plot_and_save_power_spectral_curves(n2_data, relative_n2_data, 'N2', specific_figure_path, log_scale=False)
plot_and_save_power_spectral_curves(n3_data, relative_n3_data, 'N3', specific_figure_path, log_scale=False)
plot_and_save_power_spectral_curves(n2_data, relative_n2_data, 'N2', specific_figure_path, log_scale=True)
plot_and_save_power_spectral_curves(n3_data, relative_n3_data, 'N3', specific_figure_path, log_scale=True)
