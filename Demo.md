# Demo Instructions

The demo script automates the execution of the ACT-ML pipeline, illustrating its functionality with a predefined selection of classifiers and parameters. It conducts training and validation using different classifiers and random states, computing average results over multiple runs. The demo leverages a small simulated dataset composed of randomly generated numbers.

## Running the Demo

1. **Navigate to the analysis directory**:
    ```sh
    cd analysis
    ```

2. **Run the demo script**:
    ```sh
    ./demo.sh
    ```

## Expected Output

The demo script executes the following steps:

- Iterates over each classifier and random state combination.
- Runs the training and validation for each combination, utilizing the specified features and input data.
- Saves the results from each iteration in separate CSV files: one for predictions, one for performance metrics, and one for feature importances.
- Computes the average results across multiple runs with different random states for each classifier, and outputs three CSV files: one for predictions, one for performance metrics, and one for feature importances.
  
## Run Time
