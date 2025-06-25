# axes adjusted to match the range of each pair of variables, for better visual comparison

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Configuration
variable_of_interest = 'total_N2_density_ROI_C3C4'
data_folder = '/Users/boshra/Desktop/CEAMS internship/Samantha_reports/new_reports_controls'
destination_folder = os.path.join(data_folder, 'plots')
show_regression = False  # False: hide regression lines and stats

os.makedirs(destination_folder, exist_ok=True)

def load_and_sort_data(file_path, sort_key):
    """Load data from a TSV or Excel file and sort it based on a specified key.
    Parameters:
    - file_path: Path to the file.
    - sort_key: Key to sort the data by.
    Returns:
    - DataFrame containing the sorted data.
    """
    if os.path.exists(file_path):
        if file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format for {file_path}.")
            return None
        # Convert the sort_key column to string to avoid mixed type issues
        df[sort_key] = df[sort_key].astype(str)
        df_sorted = df.sort_values(by=sort_key).reset_index(drop=True)
        return df_sorted
    else:
        print(f"File {file_path} not found.")
        return None

def process_and_plot():
    """Process data and plot the correlation between SUMO, A7, and Martin.
    Save the plot as a PDF in the destination folder.
    """
    # Load and sort data for all detectors

    # Updated SUMO file paths
    sumo_controls_file_path = os.path.join(data_folder, 'SUMO_controls.tsv')
    sumo_new_controls_file_path = os.path.join(data_folder, 'SUMO_just_new_controls.tsv')

    # Load SUMO data from both files
    df_sumo_controls = load_and_sort_data(sumo_controls_file_path, sort_key='filename')
    df_sumo_new_controls = load_and_sort_data(sumo_new_controls_file_path, sort_key='filename')

    # Load A7 and Martin data
    a7_file_path = os.path.join(data_folder, 'A7_new_controls.xlsx')
    martin_file_path = os.path.join(data_folder, 'Martin_new_controls.xlsx')
    df_a7 = load_and_sort_data(a7_file_path, sort_key='filename')
    df_martin = load_and_sort_data(martin_file_path, sort_key='filename')

    # Combine SUMO datasets first
    if df_sumo_controls is not None and df_sumo_new_controls is not None:
        # Add control_group column before combining
        df_sumo_controls['control_group'] = 'original_controls'
        df_sumo_new_controls['control_group'] = 'new_controls'

        # Concatenate first (before sorting)
        df_sumo = pd.concat([df_sumo_controls, df_sumo_new_controls], ignore_index=True)

        # Now sort the combined dataframe
        df_sumo = df_sumo.sort_values(by='filename').reset_index(drop=True)

    elif df_sumo_controls is not None:
        df_sumo_controls['control_group'] = 'original_controls'
        df_sumo = df_sumo_controls.sort_values(by='filename').reset_index(drop=True)
    elif df_sumo_new_controls is not None:
        df_sumo_new_controls['control_group'] = 'new_controls'
        df_sumo = df_sumo_new_controls.sort_values(by='filename').reset_index(drop=True)
    else:
        df_sumo = None

    if df_sumo is not None and df_a7 is not None and df_martin is not None:
        # Prepare A7 and Martin data (sort and reset index)
        df_a7 = df_a7.sort_values(by='filename').reset_index(drop=True)
        df_martin = df_martin.sort_values(by='filename').reset_index(drop=True)

        # Extract the relevant columns from SUMO (keep control_group)
        df_sumo = df_sumo[['filename', variable_of_interest, 'control_group']]
        df_a7 = df_a7[['filename', variable_of_interest]]
        df_martin = df_martin[['filename', variable_of_interest]]

        # Rename the variable_of_interest columns
        df_sumo.rename(columns={variable_of_interest: 'SUMO_value'}, inplace=True)
        df_a7.rename(columns={variable_of_interest: 'A7_value'}, inplace=True)
        df_martin.rename(columns={variable_of_interest: 'Martin_value'}, inplace=True)

        # Add index column for merging (since we reset index after sorting)
        df_sumo['merge_index'] = df_sumo.index
        df_a7['merge_index'] = df_a7.index
        df_martin['merge_index'] = df_martin.index

        # Print initial counts
        print("\nInitial counts:")
        print(f"SUMO subjects: {len(df_sumo)}")
        print(f"A7 subjects: {len(df_a7)}")
        print(f"Martin subjects: {len(df_martin)}")

        # Now merge based on the index instead of filename
        print("\nMerging SUMO and A7 based on index...")
        df_merged = pd.merge(df_sumo, df_a7, on='merge_index', how='inner')
        print(f"After merging SUMO and A7: {len(df_merged)} subjects")

        # Now merge with Martin based on index
        print("\nMerging with Martin based on index...")
        df_merged_before_martin = df_merged.copy()
        df_merged = pd.merge(df_merged, df_martin, on='merge_index', how='inner')
        print(f"After merging with Martin: {len(df_merged)} subjects")

        # Before dropping NaNs
        print("\nBefore dropping NaNs:")
        print(f"Total subjects: {len(df_merged)}")

        # Check for NaNs before dropping
        nan_count = df_merged[['SUMO_value', 'A7_value', 'Martin_value']].isna().any(axis=1).sum()
        print(f"Number of rows with any NaN in key columns: {nan_count}")

        # Create a temporary dataframe to identify subjects with NaN values
        temp_df1 = pd.merge(df_sumo, df_a7, on='merge_index', how='inner')
        temp_df_pre_nan_drop = pd.merge(temp_df1, df_martin, on='merge_index', how='inner')
        temp_df_pre_nan_drop['has_nan'] = temp_df_pre_nan_drop[['SUMO_value', 'A7_value', 'Martin_value']].isna().any(axis=1)
        subjects_with_nan = temp_df_pre_nan_drop[temp_df_pre_nan_drop['has_nan']]['filename_x']
        print("\nSubjects excluded due to NaN values in measurements:")
        print(f"Number: {len(subjects_with_nan)}")
        print("Filenames:", subjects_with_nan.tolist())

        # Also show total unique subjects in each dataset
        all_filenames_sumo = set(df_sumo['filename'])
        all_filenames_a7 = set(df_a7['filename'])
        all_filenames_martin = set(df_martin['filename'])
        all_filenames = all_filenames_sumo.union(all_filenames_a7).union(all_filenames_martin)
        print(f"\nTotal unique subjects across all datasets: {len(all_filenames)}")
        print(f"Subjects present in all three datasets at matching positions: {len(temp_df_pre_nan_drop)}")
        print(f"Subjects missing from at least one dataset or in different positions: {len(all_filenames) - len(temp_df_pre_nan_drop)}")

        # Now drop rows where any of the values are NaN
        initial_count = len(df_merged)
        df_merged.dropna(subset=['SUMO_value', 'A7_value', 'Martin_value'], inplace=True)
        final_count = len(df_merged)
        print(f"\nAfter dropping NaNs: {final_count} subjects remain ({initial_count - final_count} excluded)")


        # Define custom colors for the control groups that are lighter
        palette = {'original_controls': '#4D71A3', 'new_controls': '#FFB347'}  # Soft blue and soft orange

        plt.figure(figsize=(15, 5))

        # For SUMO vs A7 plot (first subplot):
        sumo_values = df_merged['SUMO_value']
        a7_values = df_merged['A7_value']
        global_min = min(sumo_values.min(), a7_values.min())
        global_max = max(sumo_values.max(), a7_values.max())
        range_padding = 0.1 * (global_max - global_min)
        adjusted_min = global_min - range_padding
        adjusted_max = global_max + range_padding

        plt.subplot(1, 3, 1)
        if show_regression:
            X = df_merged['SUMO_value']
            y = df_merged['A7_value']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            p_value = model.pvalues[1]

            # Plot points with different colors for each control group
            sns.scatterplot(x='SUMO_value', y='A7_value', hue='control_group',
                            data=df_merged, palette=palette, alpha=0.7)

            # Add regression line in a darker color (green)
            sns.regplot(x='SUMO_value', y='A7_value', data=df_merged, color='#2E7D32',
                        label=f'All data: R²={r_squared:.2f}, p={p_value:.2e}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(x='SUMO_value', y='A7_value', hue='control_group',
                            data=df_merged, palette=palette, alpha=0.7)

        plt.title('SUMO vs A7')
        plt.xlabel('SUMO ' + variable_of_interest)
        plt.ylabel('A7 ' + variable_of_interest)
        plt.grid(True)
        plt.xlim(adjusted_min, adjusted_max)  # Set equal x-axis limits
        plt.ylim(adjusted_min, adjusted_max)  # Set equal y-axis limits
        plt.plot([adjusted_min, adjusted_max], [adjusted_min, adjusted_max], 'k--')  # Add diagonal reference line

        # For SUMO vs Martin plot (second subplot):
        sumo_values = df_merged['SUMO_value']
        martin_values = df_merged['Martin_value']
        global_min = min(sumo_values.min(), martin_values.min())
        global_max = max(sumo_values.max(), martin_values.max())
        range_padding = 0.1 * (global_max - global_min)
        adjusted_min = global_min - range_padding
        adjusted_max = global_max + range_padding

        plt.subplot(1, 3, 2)
        if show_regression:
            X = df_merged['SUMO_value']
            y = df_merged['Martin_value']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            p_value = model.pvalues[1]

            sns.scatterplot(x='SUMO_value', y='Martin_value', hue='control_group',
                            data=df_merged, palette=palette, alpha=0.7)
            sns.regplot(x='SUMO_value', y='Martin_value', data=df_merged, color='#2E7D32',
                        label=f'All data: R²={r_squared:.2f}, p={p_value:.2e}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(x='SUMO_value', y='Martin_value', hue='control_group',
                            data=df_merged, palette=palette, alpha=0.7)

        plt.title('SUMO vs Martin')
        plt.xlabel('SUMO ' + variable_of_interest)
        plt.ylabel('Martin ' + variable_of_interest)
        plt.grid(True)
        plt.xlim(adjusted_min, adjusted_max)  # Set x-axis limits
        plt.ylim(adjusted_min, adjusted_max)  # Set y-axis limits to match x-axis
        plt.plot([adjusted_min, adjusted_max], [adjusted_min, adjusted_max], 'k--')  # Add diagonal line

        # For A7 vs Martin plot (third subplot):
        a7_values = df_merged['A7_value']
        martin_values = df_merged['Martin_value']
        global_min = min(a7_values.min(), martin_values.min())
        global_max = max(a7_values.max(), martin_values.max())
        range_padding = 0.1 * (global_max - global_min)
        adjusted_min = global_min - range_padding
        adjusted_max = global_max + range_padding

        plt.subplot(1, 3, 3)
        if show_regression:
            X = df_merged['A7_value']
            y = df_merged['Martin_value']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            p_value = model.pvalues[1]

            sns.scatterplot(x='A7_value', y='Martin_value', hue='control_group',
                            data=df_merged, palette=palette, alpha=0.7)
            sns.regplot(x='A7_value', y='Martin_value', data=df_merged, color='#2E7D32',
                        label=f'All data: R²={r_squared:.2f}, p={p_value:.2e}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(x='A7_value', y='Martin_value', hue='control_group',
                            data=df_merged, palette=palette, alpha=0.7)

        plt.title('A7 vs Martin')
        plt.xlabel('A7 ' + variable_of_interest)
        plt.ylabel('Martin ' + variable_of_interest)
        plt.grid(True)
        plt.xlim(adjusted_min, adjusted_max)  # Set x-axis limits
        plt.ylim(adjusted_min, adjusted_max)  # Set y-axis limits to match x-axis
        plt.plot([adjusted_min, adjusted_max], [adjusted_min, adjusted_max], 'k--')  # Add diagonal line

        plt.tight_layout()

        # Save the plot as a PDF file
        plot_filename = os.path.join(destination_folder, 'correlation_SUMO_A7_Martin_with_control_groups.pdf')
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved as {plot_filename}")
    else:
        print("Data could not be loaded.")

# Generate and save the plot
process_and_plot()



