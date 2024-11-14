# Author: Parisa Mapar

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
import logging

# Function to calculate feature importances based on the classifier type
def calculate_feature_importances(cf, classifier_name, features, train_set, target_train):

    try:
        if classifier_name == 'Logistic_Regression':
        
            # Use the coefficients of the best estimator for logistic regression
            imps = cf.best_estimator_.coef_[0]
            
        elif classifier_name == 'Gaussian_Naive_Bayes':
        
            # Use permutation importance for Naive Bayes
            imps = permutation_importance(cf, train_set[features], target_train).importances_mean
            
        elif classifier_name =='Random_Forest':
        
            # Use feature importances from tree-based models
            imps = cf.best_estimator_.feature_importances_
            
        return imps
        
    except Exception as e:
        # Log any errors encountered
        logging.error(f"Error in calculating feature importances: {e}")
        raise
    
    
# Function to save feature importances to a CSV file and plot them as a bar chart
def save_feature_importances(importances, features, results_path_classifier, classifier_name, title):

    try:
        # Create a DataFrame and save it to a CSV file
        imp = pd.DataFrame({'Feature': features, 'Importance': np.round(importances, 3)})
        imp.to_csv(os.path.join(results_path_classifier, f'{classifier_name}_feature_importances_{title}.csv'), index=False)

        # Sort and plot feature importances
        indices = np.argsort(importances)
        plt.figure(figsize=(7, 3))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importances')
        plt.tight_layout()
        # Save the plot as an image
        plt.savefig(os.path.join(results_path_classifier, f'{classifier_name}_feature_importances_{title}.jpeg'))
        plt.close()
        
    except Exception as e:
        # Log any errors encountered
        logging.error(f"Error in saving feature importances: {e}")
        raise




