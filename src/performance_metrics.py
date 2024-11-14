# Author: Parisa Mapar

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc, accuracy_score, f1_score
import logging

# This function calculates various performance metrics for a classification model, including precision, negative predictive value, sensitivity, specificity, accuracy, F1 score, and AUC (Area Under the Curve). It processes a given dataset containing actual and predicted values, computes these metrics, and returns them in a formatted DataFrame.

def performance_metrics(data_set):

    # Define the columns for the results DataFrame to store various performance metrics.
    results_columns = ['Precision (Positive Predictive Value)', 'Negative Predictive Value', 'Sensitivity', 'Specificity', 'Accuracy', 'F1 Score', 'AUC']
    
    # Initialize an empty DataFrame with the specified columns.
    results = pd.DataFrame(columns=results_columns)

    try:
    
        # Create binary target labels: 1 for 'Non_Responder', 0 for other responses.
        target_labels = np.where(data_set['EOT_response'] == 'Non_Responder', 1, 0)
        
        # Calculate the false positive rate, true positive rate, and thresholds for the ROC curve.
        fpr, tpr, thresholds = roc_curve(target_labels, data_set['prediction_proba_1'])
        
        # Compute the Area Under the Curve (AUC) from the ROC curve.
        aunc = auc(fpr, tpr)
        
        # Calculate precision (positive predictive value) with zero_division set to 0 to handle divisions by zero.
        precision = precision_score(target_labels, np.array(data_set['prediction'], dtype=int), zero_division=0)
        
        # Get the confusion matrix and extract true negatives (tn), false positives (fp), false negatives (fn), and true positives (tp).
        tn, fp, fn, tp = confusion_matrix(target_labels, np.array(data_set['prediction'], dtype=int)).ravel()
        
        # Calculate the negative predictive value (NPV).
        npv = tn / (tn + fn)
        
        # Calculate sensitivity (recall).
        sensitivity = recall_score(target_labels, np.array(data_set['prediction'], dtype=int))
        
        # Calculate specificity (true negative rate).
        specificity = tn / (tn + fp)
        
        # Calculate accuracy.
        accuracy = accuracy_score(target_labels, np.array(data_set['prediction'], dtype=int))
        
        # Calculate the F1 score.
        f1 = f1_score(target_labels, np.array(data_set['prediction'], dtype=int))
        
        # Store all calculated metrics in the results DataFrame, rounding them to two decimal places.
        results.loc[0, 'Precision (Positive Predictive Value)'] = round(precision, 2)
        results.loc[0, 'Negative Predictive Value'] = round(npv, 2)
        results.loc[0, 'Sensitivity'] = round(sensitivity, 2)
        results.loc[0, 'Specificity'] = round(specificity, 2)
        results.loc[0, 'Accuracy'] = round(accuracy, 2)
        results.loc[0, 'F1 Score'] = round(f1, 2)
        results.loc[0, 'AUC'] = round(aunc, 2)
        
        # Return the DataFrame containing the performance metrics.
        return results

    except Exception as e:
    
        # Log any error encountered during the metric calculation and re-raise the exception.
        logging.error(f"Error in calculating performance metrics: {e}")
        raise
        
    

