import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Configuration
variable_of_interest = 'total_N2_density_ROI_C3C4'
data_folder = '/Users/boshra/Desktop/CEAMS internship/Samantha_reports/sumo_vs_a7'
destination_folder = os.path.join(data_folder, 'plots')
show_regression = True  # False: hide regression lines and stats
use_group_info = True  # False: ignore group information from Excel

os.makedirs(destination_folder, exist_ok=True)

def load_and_sort_data(file_path, sort_key):
    """
    Load data from a TSV or Excel file and sort it based on a specified key.

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

def load_group_info(excel_path):
    """
    Load group information from an Excel file.

    Parameters:
    - excel_path: Path to the Excel file.

    Returns:
    - DataFrame containing the group information.
    """
    if os.path.exists(excel_path):
        return pd.read_excel(excel_path)
    else:
        print(f"File {excel_path} not found.")
        return None

def process_and_plot():
    """
    Process data and plot the correlation between SUMO and A7.
    Save the plot as a PDF in the destination folder.
    """
    # Load and sort data for both detectors
    sumo_file_path = os.path.join(data_folder, 'Spindles_SUMO_TOTAL.tsv')
    a7_file_path = os.path.join(data_folder, 'Spindles_A7_TOTAL.xlsx')  # Assuming Excel format for A7

    df_sumo = load_and_sort_data(sumo_file_path, sort_key='filename')
    df_a7 = load_and_sort_data(a7_file_path, sort_key='filename')

    if df_sumo is not None and df_a7 is not None:
        # Extract the relevant columns
        df_sumo = df_sumo[['filename', variable_of_interest]]
        df_a7 = df_a7[['filename', variable_of_interest]]

        # Reset indices to align the datasets based on their sorted order
        df_sumo.reset_index(drop=True, inplace=True)
        df_a7.reset_index(drop=True, inplace=True)

        # Create a new DataFrame by combining the two based on their index
        df_merged = pd.DataFrame({
            'filename_SUMO': df_sumo['filename'],
            'SUMO_value': df_sumo[variable_of_interest],
            'filename_A7': df_a7['filename'],
            'A7_value': df_a7[variable_of_interest]
        })

        # Drop rows where any of the values of interest are NaN
        df_merged.dropna(subset=['SUMO_value', 'A7_value'], inplace=True)

        if use_group_info:
            # Load group information
            excel_path = os.path.join(data_folder, 'demo_info_IH_control_Samantha.xlsx')
            df_group_info = load_group_info(excel_path)

            if df_group_info is not None:
                # Convert filename column to string to ensure consistency
                df_group_info['filename'] = df_group_info['filename'].astype(str)

                # Merge group information with the merged dataset using an outer join to retain all rows
                # Here, we'll merge with SUMO filenames, but this might need adjustment.
                df_merged = pd.merge(df_merged, df_group_info, left_on='filename_SUMO', right_on='filename', how='left')
                df_merged.drop(columns=['filename'], inplace=True)  # Drop the redundant filename column from group info

        plt.figure(figsize=(10, 6))

        if use_group_info and 'Groups (1 IH, 0 HC)' in df_merged.columns:
            # Plot with different colors for each group
            ax = sns.scatterplot(x='SUMO_value', y='A7_value',
                                 hue='Groups (1 IH, 0 HC)', data=df_merged, palette='viridis')

            if show_regression:
                # Plot regression lines for each group
                sns.lmplot(x='SUMO_value', y='A7_value',
                           hue='Groups (1 IH, 0 HC)', data=df_merged, palette='viridis', height=6, aspect=1.5)
        else:
            # Plot without group differentiation
            if show_regression:
                # Prepare data for regression
                X = df_merged['SUMO_value']
                y = df_merged['A7_value']
                X = sm.add_constant(X)  # Adds a constant term to the predictor

                # Fit linear regression model
                model = sm.OLS(y, X).fit()

                # Get R² and p-value
                r_squared = model.rsquared
                p_value = model.pvalues[1]

                # Plot the correlation with regression line
                sns.regplot(x='SUMO_value', y='A7_value',
                            data=df_merged, color='blue',
                            label=f'R²={r_squared:.2f}, p={p_value:.2e}')
            else:
                # Plot without regression line
                sns.scatterplot(x='SUMO_value', y='A7_value',
                                data=df_merged, color='blue')

        plt.title(f'Correlation between SUMO and A7')
        plt.xlabel('SUMO ' + variable_of_interest)
        plt.ylabel('A7 ' + variable_of_interest)
        plt.grid(True)
        if show_regression and not use_group_info:
            plt.legend(title='Regression Stats')

        # Save the plot as a PDF file
        plot_filename = os.path.join(destination_folder, 'correlation_SUMO_A7.pdf')
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {plot_filename}")
    else:
        print("Data could not be loaded.")

# Generate and save the plot
process_and_plot()
