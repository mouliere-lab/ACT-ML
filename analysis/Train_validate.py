# Author: Parisa Mapar

import os
import sys
import warnings
import logging
import argparse
import numpy as np

# Silence warnings
warnings.filterwarnings("ignore")

# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Ensure the parent directory is in the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.data_processing import preprocess_data
from src.load_data import load_data
from src.train import train_classifier
from src.performance_metrics import performance_metrics
from src.classifier_info import get_classifier_info
from src.feature_importance import calculate_feature_importances, save_feature_importances
from src.evaluate import test_classifier


def train_validate(features, classifier_name, input_file_path, path_to_save_results, rs, nr_jobs, cv=4):
    """
    Perform training and validation for a given classifier.

    Parameters:
    -----------
        features : list
            List of feature names.
        classifier_name : str
            Name of the classifier.
        input_file_path : str
            Path to the input data file.
        path_to_save_results : str
            Path to the directory to save the results.
        rs : int
            Random state.
        nr_jobs : int
            Number of parallel jobs.
        cv: int
            Optional, number of folds for cross validation

    Returns:
    --------
        None
        Saves CSV filse with predictions, performance metrics, and feature importances.
    """
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Classifier: {classifier_name}")
    logging.info(f"Random State: {rs}")
    
    # Get classifier parameters and create result directory
    classifier_info = get_classifier_info(classifier_name, nr_jobs)
    results_path_classifier = os.path.join(path_to_save_results, 'Results', classifier_name, f'rs_{rs}')
    os.makedirs(results_path_classifier, exist_ok=True)

    try:
        
        # Load and preprocess data
        data = preprocess_data(load_data(input_file_path))
        data = data.assign(prediction_proba_1='NaN', prediction='NaN')
            
        # Prepare training data
        train_set = data[data['split'] == 'Training'].sample(frac=1, random_state=rs).reset_index(drop=True)
        target_train = np.where(train_set['EOT_response'] == 'Non_Responder', 1, 0)

        # Train the classifier
        best_classifier = train_classifier(train_set, features, target_train, classifier_info, classifier_name, nr_jobs, cv=cv)

        # Calculate feature importances if applicable
        if classifier_name != 'SVM':
            importances = calculate_feature_importances(best_classifier, classifier_name, features, train_set, target_train)
            save_feature_importances(importances, features, results_path_classifier, classifier_name, f'rs_{rs}')    


        # Test the classifier in both training and test cohort and save predictions
        y_test_pred, y_test_proba = test_classifier(best_classifier, data, features)
        data['prediction'], data['prediction_proba_1'] = y_test_pred, y_test_proba
        data.to_csv(os.path.join(results_path_classifier, f'{classifier_name}_predictions_rs_{rs}.csv'), index=False)


        
        # Calculate and save performance metrics for validation set (timepoint 1)
        val_data = data[data['split'] != 'Training']
        data_t1_val = val_data[val_data['timepoint'] == 1]
        perf_metrics_t1_val = performance_metrics(data_t1_val)
        perf_metrics_t1_val.to_csv(os.path.join(results_path_classifier, f'{classifier_name}_performance_metrics_rs_{rs}_t1.csv'), index=False)

    except Exception as e:
        logging.error(f"Error in classification: {e}")
        raise


def main():

    parser = argparse.ArgumentParser(description='Perform training and validation for a given classifier.')
    parser.add_argument('--features', nargs='+', type=str, required=True, help='List of feature names')
    parser.add_argument('--classifier_name', type=str, required=True, help='Name of the classifier')
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--path_to_save_results', type=str, required=True, help='Path to save results')
    parser.add_argument('--rs', type=int, required=True, help='Random state')
    parser.add_argument('--nr_jobs', type=int, required=True, help='Number of parallel jobs')
    parser.add_argument('--cv', type=int, default=4, help='Number of folds for cross-validation')

    args = parser.parse_args()
    train_validate(args.features, args.classifier_name, args.input_file_path, args.path_to_save_results, args.rs, args.nr_jobs, args.cv)

if __name__ == "__main__":
    main()

# Run this script from the terminal using:
# python Train_validate.py --features features_used_for_the_analysis --classifier_name classifier_name --input_file_path path/to/your/data.xlsx --path_to_save_results path/to/save/results --rs random_state --nr_jobs number_of_cores --cv number_of_cv_folds

    
    
    
    
    
    
    
    




