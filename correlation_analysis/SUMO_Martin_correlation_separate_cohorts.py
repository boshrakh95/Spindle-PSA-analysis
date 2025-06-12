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
destination_folder = data_folder

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
    file_name = os.path.join(data_folder, file_name)
    if os.path.exists(file_name):
        return pd.read_csv(file_name, sep='\t')
    else:
        print(f"File {file_name} not found.")
        return None

def process_and_plot(group):
    """
    Process data and plot the correlation between Martin and SUMO for a given group.
    Save the plot as a PDF in the destination folder.

    Parameters:
    - group: The participant group for which to process and plot data.
    """
    # Load data for the group from both detectors
    df_martin = load_data(group, 'MARTIN')
    df_sumo = load_data(group, 'SUMO')

    if df_martin is not None and df_sumo is not None:
        # Extract the relevant columns
        df_martin = df_martin[['filename', variable_of_interest]]
        df_sumo = df_sumo[['filename', variable_of_interest]]

        # Merge the data on the 'filename' column
        df_merged = pd.merge(df_martin, df_sumo, on='filename', suffixes=('_Martin', '_SUMO'))

        # Prepare data for regression
        X = df_merged[f'{variable_of_interest}_Martin']
        y = df_merged[f'{variable_of_interest}_SUMO']
        X = sm.add_constant(X)  # Adds a constant term to the predictor

        # Fit linear regression model
        model = sm.OLS(y, X).fit()

        # Get R² and p-value
        r_squared = model.rsquared
        p_value = model.pvalues[1]

        # Plot the correlation with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(x=f'{variable_of_interest}_Martin', y=f'{variable_of_interest}_SUMO', data=df_merged)
        plt.title(f'Correlation between Martin and SUMO for {group}\nR² = {r_squared:.2f}, p-value = {p_value:.2e}')
        plt.xlabel('Martin ' + variable_of_interest)
        plt.ylabel('SUMO ' + variable_of_interest)
        plt.grid(True)

        # Save the plot as a PDF file
        plot_filename = os.path.join(destination_folder, f'correlation_{group}.pdf')
        plt.savefig(plot_filename, format='pdf')
        plt.close()
        print(f"Plot saved as {plot_filename}")
    else:
        print(f"Data for {group} could not be loaded.")

# Generate and save plots for each group
for group in groups:
    process_and_plot(group)






