# ACT Score Pipeline:

## Overview
ACT stands for Aberrations, Contribution of short fragments, and Terminal motif analyses—a machine learning pipeline designed to predict the outcomes of Diffuse Large B-Cell Lymphoma (DLBCL) patients by analyzing on-treatment plasma samples. We leveraged machine learning techniques to develop and validate a predictive model using four genomic and fragmentomic features obtained from a single run of whole-genome sequencing data. The four key features were: enhanced tumor fraction estimated by ichorCNA, the proportion of short DNA fragments (20-150bp), the Fragment-End Integrated Analysis (FrEIA) score, and the Gini Diversity Index.

To benchmark performance, we evaluated four classifiers: Random Forest, Logistic Regression, Gaussian Naive Bayes, and Support Vector Machines (SVMs). Each classifier underwent hyperparameter optimization using a 4-fold cross-validation strategy aimed at maximizing predictive accuracy. To ensure the model’s robustness, training data was perturbed using five distinct random seeds, with hyperparameter optimization repeated for each iteration. This process generated five optimal models per classifier. For the validation cohort, the final classification prediction was determined by averaging the ACT scores produced by these five predictive models.

## System Requirements

- **Operating Systems**: Windows, macOS, Linux
- **Dependencies**: Conda (latest version)
- **Tested Versions**: Python 3.12
- **Non-Standard Hardware**: None

## Hardware Requirements
The ACT-ML pipeline is designed to run on HPC clusters, as some steps can be resource-intensive with real-world datasets. However, for smaller datasets, the pipeline can also be run on standard computers.

## Installation Guide

### Prerequisites
Prior to downloading ACT-ML pipeline, users should install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

### Steps

1. **Clone the repository**:
    ```sh
    git clone https://github.com/mouliere-lab/ACT-ML.git
    cd ACT-ML
    ```
2. **Create the Conda environment**:
    ```sh
    conda env create -f ACT-ML.yml
    ```

3. **Activate the environment**:
    ```sh
    conda activate ACT-ML
    ```
4. **Verify the installation**:
    ```sh
    python --version
    ```

## Demo

The demo script automates the execution of the ACT-ML pipeline, illustrating its functionality with a predefined selection of classifiers and parameters. It conducts training and validation using different classifiers and random states, computing average results over multiple runs. The demo leverages a small simulated dataset composed of randomly generated numbers.

### Instructions to Run the Demo

1. **Navigate to the analysis directory**:
    ```sh
    cd analysis
    ```

2. **Run the demo script**:
    ```sh
    ./demo.sh
    ```

### Expected Output

The demo script executes the following steps:

- Iterates over each classifier and random state combination.
- Runs the training and validation for each combination, utilizing the specified features and input data.
- Saves the results from each iteration in separate CSV files: one for predictions, one for performance metrics, and one for feature importances.
- Computes the average results across multiple runs with different random states for each classifier, and outputs three CSV files: one for predictions, one for performance metrics, and one for feature importances.
  
## Run Time

## Instructions for Use

### Running the pipeline on your data

1. **Prepare your data**: Ensure your data format matches the expected input format. The input data needs to be an excell file compromising the following columns: 'LP', 'timepoint', 'EOT_response', and feature columns.
2. **Navigate to the analysis directory**:
    ```sh
    cd analysis
    ```
3. **Run the main scripts**:
   - Performing Leave-One-Out Cross-Validation (LOOCV) for a given classifier and for a given random state
    ```sh
    python LOOCV.py --features features_used_for_the_analysis --classifier_name classifier_name --input_file_path path/to/your/data.xlsx -- path_to_save_results path/to/save/results --rs random_state --nr_jobs number_of_cores --cv number_of_cv_folds
    ```

   - Calculate the average of performance metrics and predictions over different runs with different random states for a given classifier
    ```sh
    python average_runs.py --classifier classifier_name --results-path path/to/directory/containing/results/ --rs number_of_runs_to_be_averaged
    ```  
   - Aggregate average predictions from multiple classifiers and compute ensemble performance metrics and predictions.
    ```sh
    python ensemble.py --classifiers classifier1 classifier2 classifier3 ... classifier n --classifiers-path path/to/directory/containing/classifiers/results/ --results-path path/to/save/results
    ``` 
