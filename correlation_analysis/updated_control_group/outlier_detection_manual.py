# Script for identifying outliers in control group data based on specified thresholds for pairs of detectors.
# Specify the detectors and their thresholds for outlier detection
# This script works with visually selected thresholds. Refer to outlier_detection_automatic_dist_reg_line.py for automatic detection ...
# based on the distance from the regression line.

import pandas as pd
import os

# Configuration
variable_of_interest = 'total_N2_density_ROI_C3C4'
data_folder = '/Users/boshra/Desktop/CEAMS internship/Samantha_reports/new_reports_controls'
destination_folder = os.path.join(data_folder, 'outliers')
os.makedirs(destination_folder, exist_ok=True)

# Define detectors and thresholds as variables that can be easily modified
# None values mean that threshold is not used
detector_pairs = [
    # ('SUMO', 'A7', {'high1': 8, 'low1': None, 'high2': None, 'low2': 5})
    # ('SUMO', 'Martin', {'high1': 7.9, 'low1': None, 'high2': None, 'low2': 4})
    # ('SUMO', 'Martin', {'high1': None, 'low1': 2, 'high2': 4, 'low2': None})
    ('A7', 'Martin', {'high1': None, 'low1': 1, 'high2': 4, 'low2': None})
]
# Note: these four criteria are for "correlation_SUMO_A7_Martin_with_control_groups_without_stats_adjusted_axes_annotated.pdf"

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
        # Convert the sort_key column to string to avoid mixed type issues
        df[sort_key] = df[sort_key].astype(str)
        df_sorted = df.sort_values(by=sort_key).reset_index(drop=True)
        return df_sorted
    else:
        print(f"File {file_path} not found.")
        return None

def detect_outliers(df_pair, detector1, detector2, thresholds):
    """
    Detect outliers based on thresholds for two detectors.

    Parameters:
    - df_pair: DataFrame containing data for the detector pair
    - detector1: Name of the first detector
    - detector2: Name of the second detector
    - thresholds: Dictionary containing threshold values

    Returns:
    - A tuple containing:
        - A dataframe with outlier rows
        - A list of outlier filenames
        - A list of outlier reasons
    """
    # Initialize condition
    condition = pd.Series([True] * len(df_pair))  # Start with all True, then we'll apply AND logic

    # Build the condition based on specified thresholds
    # We'll use AND logic - all specified thresholds must be met
    if thresholds['high1'] is not None:
        condition = condition & (df_pair[f'{detector1}_value'] > thresholds['high1'])
    if thresholds['low1'] is not None:
        condition = condition & (df_pair[f'{detector1}_value'] < thresholds['low1'])
    if thresholds['high2'] is not None:
        condition = condition & (df_pair[f'{detector2}_value'] > thresholds['high2'])
    if thresholds['low2'] is not None:
        condition = condition & (df_pair[f'{detector2}_value'] < thresholds['low2'])

    # Filter the dataframe based on the condition
    outliers = df_pair[condition]

    # Generate reasons for each outlier
    reasons = []
    for idx, row in outliers.iterrows():
        reason_parts = []
        val1 = row[f'{detector1}_value']
        val2 = row[f'{detector2}_value']

        if thresholds['high1'] is not None and val1 > thresholds['high1']:
            reason_parts.append(f"{detector1}={val1:.2f} > {thresholds['high1']}")
        if thresholds['low1'] is not None and val1 < thresholds['low1']:
            reason_parts.append(f"{detector1}={val1:.2f} < {thresholds['low1']}")
        if thresholds['high2'] is not None and val2 > thresholds['high2']:
            reason_parts.append(f"{detector2}={val2:.2f} > {thresholds['high2']}")
        if thresholds['low2'] is not None and val2 < thresholds['low2']:
            reason_parts.append(f"{detector2}={val2:.2f} < {thresholds['low2']}")

        reasons.append(" AND ".join(reason_parts) if reason_parts else "Unknown reason")

    # Get the filenames
    outlier_filenames = outliers['filename_x'].tolist()

    return outliers, outlier_filenames, reasons

def process_and_find_outliers():
    """Process data and find outliers between detector pairs."""
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
        df_merged_sumo_a7 = pd.merge(df_sumo, df_a7, on='merge_index', how='inner')
        print(f"After merging SUMO and A7: {len(df_merged_sumo_a7)} subjects")

        # Now merge with Martin based on index
        print("\nMerging with Martin based on index...")
        df_merged = pd.merge(df_merged_sumo_a7, df_martin, on='merge_index', how='inner')
        print(f"After merging with Martin: {len(df_merged)} subjects")

        # Before dropping NaNs
        print("\nBefore dropping NaNs:")
        print(f"Total subjects: {len(df_merged)}")

        # Check for NaNs before dropping
        nan_count = df_merged[['SUMO_value', 'A7_value', 'Martin_value']].isna().any(axis=1).sum()
        print(f"Number of rows with any NaN in key columns: {nan_count}")

        # Create a temporary dataframe to identify subjects with NaN values
        temp_df_pre_nan_drop = pd.merge(
            pd.merge(df_sumo, df_a7, on='merge_index', how='inner'),
            df_martin, on='merge_index', how='inner'
        )
        temp_df_pre_nan_drop['has_nan'] = temp_df_pre_nan_drop[
            ['SUMO_value', 'A7_value', 'Martin_value']
        ].isna().any(axis=1)
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

        # Create a dictionary to store all outliers information
        all_outliers_info = {}

        # Now process each detector pair to find outliers
        for detector1, detector2, thresholds in detector_pairs:
            print(f"\nProcessing pair: {detector1} vs {detector2}")

            # Prepare the dataframe for this detector pair
            if {detector1, detector2} == {'SUMO', 'A7'}:
                df_pair = df_merged[['filename_x', 'SUMO_value', 'A7_value']].copy()
                df_pair = df_pair.rename(columns={
                    'filename_x': 'filename_x',
                    'SUMO_value': 'SUMO_value',
                    'A7_value': 'A7_value'
                })
            elif {detector1, detector2} == {'SUMO', 'Martin'}:
                df_pair = df_merged[['filename_x', 'SUMO_value', 'Martin_value']].copy()
                df_pair = df_pair.rename(columns={
                    'filename_x': 'filename_x',
                    'SUMO_value': 'SUMO_value',
                    'Martin_value': 'Martin_value'
                })
            elif {detector1, detector2} == {'A7', 'Martin'}:
                # For A7 vs Martin, create a special dataframe
                df_a7_martin = pd.merge(
                    df_a7.assign(merge_index=df_a7.index),
                    df_martin.assign(merge_index=df_martin.index),
                    on='merge_index', how='inner'
                )
                df_pair = df_a7_martin[['filename_x', 'A7_value', 'Martin_value']].copy()
            else:
                print(f"Unexpected detector pair: {detector1}, {detector2}")
                continue

            # Rename columns to match the expected pattern (e.g., 'SUMO_value' to 'detector1_value')
            # So that detect_outliers function can work generically
            df_pair = df_pair.rename(columns={
                f'{detector1}_value': f'{detector1}_value',
                f'{detector2}_value': f'{detector2}_value'
            })

            # Check if there are any thresholds defined for this pair
            if any(v is not None for v in thresholds.values()):
                # Call the detect_outliers function
                outliers, outlier_filenames, reasons = detect_outliers(df_pair, detector1, detector2, thresholds)

                # Print and save the results
                print(f"\nFound {len(outliers)} outliers for {detector1} vs {detector2} with thresholds:")
                print(f"  {detector1} high threshold: {thresholds['high1']}")
                print(f"  {detector1} low threshold: {thresholds['low1']}")
                print(f"  {detector2} high threshold: {thresholds['high2']}")
                print(f"  {detector2} low threshold: {thresholds['low2']}")

                if len(outliers) > 0:
                    print("Outlier filenames:", outlier_filenames)

                    # Create a dataframe for the outlier info
                    outlier_info = []
                    for idx, row in outliers.iterrows():
                        filename = row['filename_x']
                        val1 = row[f'{detector1}_value']
                        val2 = row[f'{detector2}_value']
                        # The reasons are already prepared in the detect_outliers function
                        # We'll use the corresponding reason from the reasons list
                        # Since outliers and reasons are aligned by index
                        idx_in_outliers = outliers.index.get_loc(idx)
                        reason = reasons[idx_in_outliers]

                        outlier_info.append({
                            'filename': filename,
                            f'{detector1}_value': val1,
                            f'{detector2}_value': val2,
                            'reason': reason
                        })

                    # Create a dataframe for the outlier info and save to CSV
                    outliers_df = pd.DataFrame(outlier_info)
                    output_filename = os.path.join(
                        destination_folder,
                        f'outliers_{detector1}_vs_{detector2}.csv'
                    )
                    outliers_df.to_csv(output_filename, index=False)
                    print(f"Outlier details saved to {output_filename}")

                    # Store this information in our all_outliers_info dictionary
                    all_outliers_info[f'{detector1}_vs_{detector2}'] = {
                        'outliers': outliers_df,
                        'count': len(outliers_df)
                    }
                else:
                    print("No outliers found for this pair with the given thresholds.")
            else:
                print(f"No thresholds defined for {detector1} vs {detector2}, skipping outlier detection.")

        # After processing all detector pairs, create a summary file with all outliers
        print("\nCreating summary of all outliers found:")
        all_outliers = []
        total_outliers = 0

        # Go through all detector pairs and collect outliers
        for pair_info in detector_pairs:
            detector1, detector2, _ = pair_info
            key = f'{detector1}_vs_{detector2}'
            if key in all_outliers_info:
                pair_outliers = all_outliers_info[key]['outliers']
                pair_count = all_outliers_info[key]['count']
                total_outliers += pair_count

                # Add some information about each pair to the summary
                all_outliers.append({
                    'detector_pair': f'{detector1} vs {detector2}',
                    'count': pair_count,
                    'details': f"See {key}.csv for details"
                })

        # Add summary information
        summary_info = {
            'total_outliers_found': total_outliers,
            'detector_pairs_analyzed': [f"{d1} vs {d2}" for d1, d2, _ in detector_pairs],
            'thresholds_used': [
                {
                    'pair': f"{d1} vs {d2}",
                    'thresholds': t
                } for d1, d2, t in detector_pairs
            ]
        }

        # Save the summary to a file
        summary_filename = os.path.join(destination_folder, 'outliers_summary.csv')
        pd.DataFrame(all_outliers).to_csv(summary_filename, index=False)
        print(f"\nSummary of all outliers saved to {summary_filename}")

        # Also save the full summary info as a JSON file for complete reference
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