import pandas as pd
import numpy as np

# Load the TSV file
file_path = '/Users/boshra/Downloads/karine_PSA_power_nettoye_transposed.tsv'
data = pd.read_csv(file_path, sep='\t')

# Extract relevant columns for N2 and N3 stages
n2_columns = [col for col in data.columns if 'N2_act_abs_ROI_C3C4' in col]
n3_columns = [col for col in data.columns if 'N3_act_abs_ROI_C3C4' in col]

# Filter out subjects without N3 data
data = data.dropna(subset=n3_columns, how='all')

# Extract the group information
groups = data['Groups (IH-1 controls=0)']

# Extract relevant columns for the frequency range of 6 to 11 Hz in N3 stage
frequency_bins_of_interest = [f'total_N3_act_abs_ROI_C3C4_{freq}_{freq+1}' for freq in range(1, 11)]

# Extract the data for these frequency bins
n3_data_of_interest = data[frequency_bins_of_interest]

# Separate the data by group
control_group_data = n3_data_of_interest[groups == 0]
ih_group_data = n3_data_of_interest[groups == 1]

# Calculate mean and standard deviation for each frequency bin in each group
control_group_stats = control_group_data.describe()
ih_group_stats = ih_group_data.describe()

print("Control Group Statistics:")
print(control_group_stats)
print("\nIH Group Statistics:")
print(ih_group_stats)

##############

# Extract the absolute power data for N2 and N3
n2_data = data[n2_columns]
n3_data = data[n3_columns]

# Filter data for the Control group
control_group_data = n3_data[groups == 0]

# Calculate mean and standard deviation for each frequency bin in the Control group
control_group_mean = control_group_data.mean(axis=0)
control_group_std = control_group_data.std(axis=0)

# Calculate the min and max values for the shaded area
control_group_min = control_group_mean - control_group_std
control_group_max = control_group_mean + control_group_std

# Apply log scale to the min and max values
control_group_min_log = np.log10(control_group_min)
control_group_max_log = np.log10(control_group_max)

# Create a DataFrame to display the results
results = pd.DataFrame({
    'Frequency Bin': [col.split('_')[-2] for col in n3_columns],
    'Mean': control_group_mean,
    'Std': control_group_std,
    'Min': control_group_min,
    'Max': control_group_max,
    'Min (Log Scale)': control_group_min_log,
    'Max (Log Scale)': control_group_max_log
})

print(results)




