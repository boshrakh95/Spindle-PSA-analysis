import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Configuration
groups = ['NT1', 'NT2', 'IHobjective', 'IHsubjective', 'Controls']
detectors = ['MARTIN', 'SUMO']
variable_of_interest = 'total_N2_density_ROI_C3C4'
data_folder = '/Users/boshra/Desktop/CEAMS internship/Samantha_reports/'
destination_folder = os.path.join(data_folder, 'plots')
show_regression = True  # False: hide regression lines and stats

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

def load_data(group, detector):
    """
    Load data from a TSV file based on group and detector.

    Parameters:
    - group: The participant group.
    - detector: The spindle detector (MARTIN or SUMO).

    Returns:
    - DataFrame containing the loaded data.
    """
    file_name = f"Spindles_{detector}_{group}_TOTAL.tsv"
    file_path = os.path.join(data_folder, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep='\t')
    else:
        print(f"File {file_path} not found.")
        return None

def process_and_plot_all_groups():
    """
    Process data and plot the correlation between Martin and SUMO for all groups overlaid.
    Save the plot as a PDF in the destination folder.
    """
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, group in enumerate(groups):
        df_martin = load_data(group, 'MARTIN')
        df_sumo = load_data(group, 'SUMO')

        if df_martin is not None and df_sumo is not None:
            df_martin = df_martin[['filename', variable_of_interest]]
            df_sumo = df_sumo[['filename', variable_of_interest]]

            df_merged = pd.merge(df_martin, df_sumo, on='filename', suffixes=('_Martin', '_SUMO'))

            if show_regression:
                # Prepare data for regression
                X = df_merged[f'{variable_of_interest}_Martin']
                y = df_merged[f'{variable_of_interest}_SUMO']
                X = sm.add_constant(X)  # Adds a constant term to the predictor

                # Fit linear regression model
                model = sm.OLS(y, X).fit()

                # Get R² and p-value
                r_squared = model.rsquared
                p_value = model.pvalues[1]

                # Plot the correlation with regression line for each group
                sns.regplot(x=f'{variable_of_interest}_Martin', y=f'{variable_of_interest}_SUMO',
                            data=df_merged, color=colors[idx],
                            label=f'{group} (R²={r_squared:.2f}, p={p_value:.2e})',
                            scatter_kws={'alpha': 0.5})
            else:
                # Plot without regression line
                sns.scatterplot(x=f'{variable_of_interest}_Martin', y=f'{variable_of_interest}_SUMO',
                                data=df_merged, color=colors[idx], label=group, alpha=0.5)

    plt.title(f'Correlation between Martin and SUMO for All Groups')
    plt.xlabel('Martin ' + variable_of_interest)
    plt.ylabel('SUMO ' + variable_of_interest)
    plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Save the plot as a PDF file
    plot_filename = os.path.join(destination_folder, 'correlation_all_groups.pdf')
    plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {plot_filename}")

# Generate and save the overlay plot for all groups
process_and_plot_all_groups()
