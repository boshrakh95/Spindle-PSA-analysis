import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

# Configuration
variable_of_interest = 'total_N2_density_ROI_C3C4'
frequency_band_column = 'total_N2_act_rel_ROI_C3C4_11_16'
# frequency_band_column = 'total_N2_act_rel_ROI_C3C4_8_12'
band_name = 'Sigma'
# band_name = 'Alpha'
data_folder = '/Users/boshra/Desktop/CEAMS internship/Samantha_reports/new_reports_controls'
destination_folder = os.path.join(data_folder, 'plots')
show_regression = True

os.makedirs(destination_folder, exist_ok=True)

def load_and_sort_data(file_path, sort_key):
    """Load data from a TSV or Excel file and sort it based on a specified key."""
    if os.path.exists(file_path):
        if file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format for {file_path}.")
            return None
        df[sort_key] = df[sort_key].astype(str)
        df_sorted = df.sort_values(by=sort_key).reset_index(drop=True)
        return df_sorted
    else:
        print(f"File {file_path} not found.")
        return None

def process_and_plot():
    """Process data and plot the correlation between PSA power and N2 density for SUMO, A7, and Martin."""
    # Load and sort data for all detectors and PSA
    sumo_controls_file_path = os.path.join(data_folder, 'SUMO_controls.tsv')
    sumo_new_controls_file_path = os.path.join(data_folder, 'SUMO_just_new_controls.tsv')
    a7_file_path = os.path.join(data_folder, 'A7_new_controls.xlsx')
    martin_file_path = os.path.join(data_folder, 'Martin_new_controls.xlsx')
    psa_file_path = os.path.join(data_folder, 'PSA_TOTAL_New controls_transposed.xlsx')

    df_sumo_controls = load_and_sort_data(sumo_controls_file_path, sort_key='filename')
    df_sumo_new_controls = load_and_sort_data(sumo_new_controls_file_path, sort_key='filename')
    df_a7 = load_and_sort_data(a7_file_path, sort_key='filename')
    df_martin = load_and_sort_data(martin_file_path, sort_key='filename')
    df_psa = load_and_sort_data(psa_file_path, sort_key='filename')

    # Combine SUMO datasets
    if df_sumo_controls is not None and df_sumo_new_controls is not None:
        df_sumo_controls['control_group'] = 'original_controls'
        df_sumo_new_controls['control_group'] = 'new_controls'
        df_sumo = pd.concat([df_sumo_controls, df_sumo_new_controls], ignore_index=True)
        df_sumo = df_sumo.sort_values(by='filename').reset_index(drop=True)
    elif df_sumo_controls is not None:
        df_sumo_controls['control_group'] = 'original_controls'
        df_sumo = df_sumo_controls.sort_values(by='filename').reset_index(drop=True)
    elif df_sumo_new_controls is not None:
        df_sumo_new_controls['control_group'] = 'new_controls'
        df_sumo = df_sumo_new_controls.sort_values(by='filename').reset_index(drop=True)
    else:
        df_sumo = None

    if df_sumo is not None and df_a7 is not None and df_martin is not None and df_psa is not None:
        df_a7 = df_a7.sort_values(by='filename').reset_index(drop=True)
        df_martin = df_martin.sort_values(by='filename').reset_index(drop=True)
        df_psa = df_psa.sort_values(by='filename').reset_index(drop=True)

        df_sumo = df_sumo[['filename', variable_of_interest, 'control_group']]
        df_a7 = df_a7[['filename', variable_of_interest]]
        df_martin = df_martin[['filename', variable_of_interest]]
        df_psa = df_psa[['filename', frequency_band_column]]

        df_sumo.rename(columns={variable_of_interest: 'SUMO_value'}, inplace=True)
        df_a7.rename(columns={variable_of_interest: 'A7_value'}, inplace=True)
        df_martin.rename(columns={variable_of_interest: 'Martin_value'}, inplace=True)

        df_sumo['merge_index'] = df_sumo.index
        df_a7['merge_index'] = df_a7.index
        df_martin['merge_index'] = df_martin.index
        df_psa['merge_index'] = df_psa.index

        # df_merged = pd.merge(df_sumo, df_psa, on='merge_index', how='inner')
        # df_merged = pd.merge(df_merged, df_a7, on='merge_index', how='inner')
        # df_merged = pd.merge(df_merged, df_martin, on='merge_index', how='inner')

        # Merge SUMO and PSA data
        df_merged = pd.merge(df_sumo, df_psa, on='merge_index', how='inner', suffixes=('_sumo', '_psa'))
        # Merge the result with A7 data
        df_merged = pd.merge(df_merged, df_a7, on='merge_index', how='inner', suffixes=('', '_a7'))
        # Merge the result with Martin data
        df_merged = pd.merge(df_merged, df_martin, on='merge_index', how='inner', suffixes=('', '_martin'))

        df_merged.dropna(subset=['SUMO_value', 'A7_value', 'Martin_value', frequency_band_column], inplace=True)

        plt.figure(figsize=(18, 6))

        # Plot 1: PSA Power vs SUMO N2 Density
        plt.subplot(1, 3, 1)
        if show_regression:
            X = df_merged[frequency_band_column]
            y = df_merged['SUMO_value']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            p_value = model.pvalues[1]
            sns.scatterplot(x=frequency_band_column, y='SUMO_value', data=df_merged, color='#4D71A3', alpha=0.7)
            sns.regplot(x=frequency_band_column, y='SUMO_value', data=df_merged, color='#2E7D32', label=f'All data: R²={r_squared:.2f}, p={p_value:.2e}')
            plt.legend()
        else:
            sns.scatterplot(x=frequency_band_column, y='SUMO_value', data=df_merged, color='#4D71A3', alpha=0.7)
        plt.title(f'{band_name} Relative Power vs SUMO N2 Density')
        plt.xlabel(f'{band_name} Power')
        plt.ylabel('SUMO N2 Density')
        plt.grid(True)

        # Plot 2: PSA Power vs A7 N2 Density
        plt.subplot(1, 3, 2)
        if show_regression:
            X = df_merged[frequency_band_column]
            y = df_merged['A7_value']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            p_value = model.pvalues[1]
            sns.scatterplot(x=frequency_band_column, y='A7_value', data=df_merged, color='#4D71A3', alpha=0.7)
            sns.regplot(x=frequency_band_column, y='A7_value', data=df_merged, color='#2E7D32', label=f'All data: R²={r_squared:.2f}, p={p_value:.2e}')
            plt.legend()
        else:
            sns.scatterplot(x=frequency_band_column, y='A7_value', data=df_merged, color='#4D71A3', alpha=0.7)
        plt.title(f'{band_name} Relative Power vs A7 N2 Density')
        plt.xlabel(f'{band_name} Power')
        plt.ylabel('A7 N2 Density')
        plt.grid(True)

        # Plot 3: PSA Power vs Martin N2 Density
        plt.subplot(1, 3, 3)
        if show_regression:
            X = df_merged[frequency_band_column]
            y = df_merged['Martin_value']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared = model.rsquared
            p_value = model.pvalues[1]
            sns.scatterplot(x=frequency_band_column, y='Martin_value', data=df_merged, color='#4D71A3', alpha=0.7)
            sns.regplot(x=frequency_band_column, y='Martin_value', data=df_merged, color='#2E7D32', label=f'All data: R²={r_squared:.2f}, p={p_value:.2e}')
            plt.legend()
        else:
            sns.scatterplot(x=frequency_band_column, y='Martin_value', data=df_merged, color='#4D71A3', alpha=0.7)
        plt.title(f'{band_name} Relative Power vs Martin N2 Density')
        plt.xlabel(f'{band_name} Power')
        plt.ylabel('Martin N2 Density')
        plt.grid(True)

        plt.tight_layout()

        plot_filename = os.path.join(destination_folder, f'correlation_{band_name.lower()}_rel_power_n2_density.pdf')
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved as {plot_filename}")
    else:
        print("Data could not be loaded.")

process_and_plot()
