# Author: Parisa Mapar

import os
import sys
import logging
import numpy as np
import pandas as pd
import argparse

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Ensure the parent directory is in the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from src.performance_metrics import performance_metrics

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def average_results(classifier_name, results_path, random_states):
    """
    Calculate the average of performance metrics over different runs with different random states.
    
    Parameters:
    -----------
        classifier_name (str): Name of the classifier to analyze.
        results_path (str): Path to the directory containing the results data for different random states.
        random_states (int): Number of random state iterations to consider for averaging.
    
    Returns:
    --------
        None
        Saves an aggregated CSV file with average predictions and a CSV file with calculated performance metrics.
    """
    logging.info(f"Classifier: {classifier_name}")


    # Define paths
    results_path_classifier = os.path.join(results_path, 'Results', classifier_name)
    results_path_classifier_avg = os.path.join(results_path_classifier, 'Average')
    os.makedirs(results_path_classifier_avg, exist_ok=True)

    all_results = []

    # Read and aggregate results over different random states
    for i in range(1, random_states + 1):
        file_path = os.path.join(results_path_classifier, f'rs_{i}', f'{classifier_name}_predictions_rs_{i}.csv')

        try:
            data = pd.read_csv(file_path)
            all_results.append(data)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            logging.warning(f"File is empty: {file_path}")
        except pd.errors.ParserError:
            logging.error(f"Error parsing file: {file_path}.")
        except Exception as e:
            logging.error(f"Unexpected error with file {file_path}: {e}")

    if not all_results:
        logging.error(f"No valid data found for classifier: {classifier_name}.")
        return

    
    # Concatenate all results and compute averages
    all_results = pd.concat(all_results)

    # Define the columns that should be averaged and those that should remain unchanged
    mean_columns = ['prediction_proba_1']
    fixed_columns = [col for col in all_results.columns if col not in mean_columns and col != 'LP']

    # Create the aggregation dictionary
    agg_dict = {col: 'mean' for col in mean_columns}
    agg_dict.update({col: 'first' for col in fixed_columns})

    # Group by 'LP' and apply the aggregation functions
    avg_data = all_results.groupby('LP').agg(agg_dict).reset_index()

            
    # Apply threshold to generate final predictions
    avg_data['prediction'] = np.where(avg_data['prediction_proba_1'] >= 0.5, 1, 0)

    # Save averaged predictions
    avg_file_path = os.path.join(results_path_classifier_avg, f'{classifier_name}_average_predictions.csv')
    avg_data.to_csv(avg_file_path, index=False)
    logging.info(f"Saved averaged predictions to {avg_file_path}")

    # Compute and save performance metrics for validation sets
    val_data = avg_data[avg_data['split'] != 'Training']

    # Timepoint 1
    data_t1_val=val_data[val_data['timepoint'] == 1]
    perf_metrics_t1_val = performance_metrics(data_t1_val)
    perf_metrics_t1_file_path = os.path.join(results_path_classifier_avg, f'{classifier_name}_average_performance_metrics_t1.csv')
    perf_metrics_t1_val.to_csv(perf_metrics_t1_file_path, index=False)
    logging.info(f"Saved performance metrics for Timepoint 1 to {perf_metrics_t1_file_path}")

def main():
   
    parser = argparse.ArgumentParser(description='Calculate the average performance metrics over different runs.')
    parser.add_argument('--classifier', type=str, help='Name of the classifier')
    parser.add_argument('--results-path', type=str, help='Path to the directory containing the results')
    parser.add_argument('--rs', type=int, help='Number of random states')

    args = parser.parse_args()

    average_results(args.classifier, args.results_path, args.rs)


if __name__ == "__main__":
    main()


# To run this script in the terminal, use the following command:
# python average_runs.py --classifier classifier_name --results-path path/to/directory/containing/results/ --rs number_of_runs_to_be_averaged
