import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import numpy as np

# Configuration
variable_of_interest = 'total_N2_density_ROI_C3C4'
data_folder = '/Users/boshra/Desktop/CEAMS internship/Samantha_reports/new_reports_controls'
destination_folder = os.path.join(data_folder, 'automatic_outliers')
distance_threshold = 1.5  # Threshold for outlier detection based on Euclidean distance

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

def calculate_euclidean_distance(x, y, slope, intercept):
    """Calculate the Euclidean distance from points to the regression line."""
    numerator = np.abs(slope * x - y + intercept)
    denominator = np.sqrt(slope**2 + 1)
    return numerator / denominator

def detect_outliers(df_pair, detector1, detector2):
    """Detect outliers based on Euclidean distance from the regression line."""
    X = df_pair[[f'{detector1}_value']]
    y = df_pair[f'{detector2}_value']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    slope = model.params[1]
    intercept = model.params[0]
    distances = calculate_euclidean_distance(df_pair[f'{detector1}_value'], df_pair[f'{detector2}_value'], slope, intercept)
    df_pair['distance'] = distances
    df_pair['outlier'] = df_pair['distance'] > distance_threshold

    outliers = df_pair[df_pair['outlier']]
    outlier_filenames = outliers['filename_x'].tolist()
    reasons = [f"Distance from regression line: {distance:.2f} > {distance_threshold}" for distance in outliers['distance']]

    return outliers, outlier_filenames, reasons

def process_and_find_outliers():
    """Process data and find outliers between detector pairs."""
    # Load and sort data for all detectors
    sumo_controls_file_path = os.path.join(data_folder, 'SUMO_controls.tsv')
    sumo_new_controls_file_path = os.path.join(data_folder, 'SUMO_just_new_controls.tsv')
    a7_file_path = os.path.join(data_folder, 'A7_new_controls.xlsx')
    martin_file_path = os.path.join(data_folder, 'Martin_new_controls.xlsx')

    df_sumo_controls = load_and_sort_data(sumo_controls_file_path, sort_key='filename')
    df_sumo_new_controls = load_and_sort_data(sumo_new_controls_file_path, sort_key='filename')
    df_a7 = load_and_sort_data(a7_file_path, sort_key='filename')
    df_martin = load_and_sort_data(martin_file_path, sort_key='filename')

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

    if df_sumo is not None and df_a7 is not None and df_martin is not None:
        df_a7 = df_a7.sort_values(by='filename').reset_index(drop=True)
        df_martin = df_martin.sort_values(by='filename').reset_index(drop=True)

        df_sumo = df_sumo[['filename', variable_of_interest, 'control_group']]
        df_a7 = df_a7[['filename', variable_of_interest]]
        df_martin = df_martin[['filename', variable_of_interest]]

        df_sumo.rename(columns={variable_of_interest: 'SUMO_value'}, inplace=True)
        df_a7.rename(columns={variable_of_interest: 'A7_value'}, inplace=True)
        df_martin.rename(columns={variable_of_interest: 'Martin_value'}, inplace=True)

        df_sumo['merge_index'] = df_sumo.index
        df_a7['merge_index'] = df_a7.index
        df_martin['merge_index'] = df_martin.index

        df_merged_sumo_a7 = pd.merge(df_sumo, df_a7, on='merge_index', how='inner')
        df_merged = pd.merge(df_merged_sumo_a7, df_martin, on='merge_index', how='inner')

        df_merged.dropna(subset=['SUMO_value', 'A7_value', 'Martin_value'], inplace=True)

        # Create a dictionary to store all outliers information
        all_outliers_info = {}

        # Process each detector pair to find outliers
        detector_pairs = [('SUMO', 'A7'), ('SUMO', 'Martin'), ('A7', 'Martin')]

        for detector1, detector2 in detector_pairs:
            print(f"\nProcessing pair: {detector1} vs {detector2}")
            if {detector1, detector2} == {'SUMO', 'A7'}:
                df_pair = df_merged[['filename', 'SUMO_value', 'A7_value']].copy()
                df_pair = df_pair.rename(columns={'filename': 'filename_x', 'SUMO_value': 'SUMO_value', 'A7_value': 'A7_value'})
            elif {detector1, detector2} == {'SUMO', 'Martin'}:
                df_pair = df_merged[['filename', 'SUMO_value', 'Martin_value']].copy()
                df_pair = df_pair.rename(columns={'filename': 'filename_x', 'SUMO_value': 'SUMO_value', 'Martin_value': 'Martin_value'})
            elif {detector1, detector2} == {'A7', 'Martin'}:
                df_pair = pd.merge(
                    df_a7.assign(merge_index=df_a7.index),
                    df_martin.assign(merge_index=df_martin.index),
                    on='merge_index', how='inner'
                )
                df_pair = df_pair[['filename_x', 'A7_value', 'Martin_value']].copy()
                df_pair = df_pair.rename(columns={'A7_value': 'A7_value', 'Martin_value': 'Martin_value'})
            else:
                print(f"Unexpected detector pair: {detector1}, {detector2}")
                continue

            df_pair = df_pair.rename(columns={
                f'{detector1}_value': f'{detector1}_value',
                f'{detector2}_value': f'{detector2}_value'
            })

            outliers, outlier_filenames, reasons = detect_outliers(df_pair, detector1, detector2)

            print(f"\nFound {len(outliers)} outliers for {detector1} vs {detector2} with distance threshold: {distance_threshold}")
            if len(outliers) > 0:
                print("Outlier filenames:", outlier_filenames)
                outlier_info = []
                for idx, row in outliers.iterrows():
                    filename = row['filename_x']
                    val1 = row[f'{detector1}_value']
                    val2 = row[f'{detector2}_value']
                    reason = reasons[outliers.index.get_loc(idx)]
                    outlier_info.append({
                        'filename': filename,
                        f'{detector1}_value': val1,
                        f'{detector2}_value': val2,
                        'reason': reason
                    })

                outliers_df = pd.DataFrame(outlier_info)
                output_filename = os.path.join(
                    destination_folder,
                    f'outliers_{detector1}_vs_{detector2}.csv'
                )
                outliers_df.to_csv(output_filename, index=False)
                print(f"Outlier details saved to {output_filename}")

                all_outliers_info[f'{detector1}_vs_{detector2}'] = {
                    'outliers': outliers_df,
                    'count': len(outliers_df)
                }

                # Plot the data with outliers and non-outliers in different colors
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=f'{detector1}_value', y=f'{detector2}_value', hue='outlier', data=df_pair,
                                palette={True: 'red', False: 'blue'}, alpha=0.7)

                # Fit regression model
                X = df_pair[[f'{detector1}_value']]
                y = df_pair[f'{detector2}_value']
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                # Plot regression line manually
                plt.plot(df_pair[f'{detector1}_value'], model.predict(X), color='green',
                         label=f'Distance Threshold: {distance_threshold}')

                plt.title(f'{detector1} vs {detector2} with Outliers')
                plt.xlabel(f'{detector1} Value')
                plt.ylabel(f'{detector2} Value')
                plt.grid(True)
                plt.legend(title='Outliers', loc='upper right')

                plot_filename = os.path.join(destination_folder, f'outliers_plot_{detector1}_vs_{detector2}.pdf')
                plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
                plt.close()
                print(f"Plot saved as {plot_filename}")

            else:
                print("No outliers found for this pair with the given distance threshold.")

        # Create a summary of all outliers found
        print("\nCreating summary of all outliers found:")
        all_outliers = []
        total_outliers = 0
        for pair_info in detector_pairs:
            detector1, detector2 = pair_info
            key = f'{detector1}_vs_{detector2}'
            if key in all_outliers_info:
                pair_outliers = all_outliers_info[key]['outliers']
                pair_count = all_outliers_info[key]['count']
                total_outliers += pair_count
                all_outliers.append({
                    'detector_pair': f'{detector1} vs {detector2}',
                    'count': pair_count,
                    'details': f"See {key}.csv for details"
                })

        summary_info = {
            'total_outliers_found': total_outliers,
            'detector_pairs_analyzed': [f"{d1} vs {d2}" for d1, d2 in detector_pairs],
            'distance_threshold': distance_threshold
        }

        # Save the summary to a file
        summary_filename = os.path.join(destination_folder, 'outliers_summary.csv')
        pd.DataFrame(all_outliers).to_csv(summary_filename, index=False)
        print(f"\nSummary of all outliers saved to {summary_filename}")

        # Save the full summary info as a JSON file for complete reference
        summary_json_filename = os.path.join(destination_folder, 'outliers_summary_complete.json')
        try:
            import json
            with open(summary_json_filename, 'w') as f:
                json.dump(summary_info, f, indent=2)
            print(f"Complete summary with all details saved to {summary_json_filename}")
        except ImportError:
            print("Could not save JSON summary (json module not available)")

    else:
        print("Data could not be loaded.")

# Generate and process the data to find outliers
process_and_find_outliers()



