#!/bin/bash

# Author: Parisa Mapar

# Demo Script: ACT-ML Pipeline Analysis
# Description: This script runs the ACT-ML pipeline analysis on a small simulated dataset composed of randomly generated numbers. It conducts training and validation using different classifiers and random states, and computes the average results over multiple runs.


# Define list of classifiers and features
classifiers=('Gaussian_Naive_Bayes' 'Logistic_Regression' 'Random_Forest' 'SVM')
features=('TF_ichorCNA_sizeSel_P20_150' 'P20_150' 'FrEIA' 'Gini')

# Get current directory
parent_dir=$(dirname "$(pwd)")

# Loop over each classifier
for classifier_name in "${classifiers[@]}"
do
    # Loop over random states
    for rs in {1..5}
    do
        echo "Classifier name: $classifier_name"
        echo "Random state: $rs"

        # Run LOOCV.py (training the classifier)
        python Train_validate.py --features "${features[@]}" --classifier_name "$classifier_name" --input_file_path "$parent_dir/data/dummy_data.xlsx" --path_to_save_results "$parent_dir/Demo_results" --rs "$rs" --nr_jobs 16 --cv 3
    done

    echo "Calculating average results for classifier: $classifier_name"
    
    # Run average_runs.py
    python average_runs.py --classifier "$classifier_name" --results-path "$parent_dir/Demo_results" --rs 5
done
