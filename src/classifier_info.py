# Author: Parisa Mapar

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# Function to get classifier information and parameter grid for hyperparameter tuning
def get_classifier_info(classifier_name,nr_jobs):

    # Dictionary containing models and their respective hyperparameter grids for tuning
    classifiers = {
    
    # Random Forest model with a specified number of parallel jobs
    'Random_Forest': {'model': RandomForestClassifier(random_state=42,n_jobs=nr_jobs),
    'params': {
        'n_estimators': [30,50, 100,200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced'],
        'oob_score':[False],
        'max_features':['sqrt', None],
        'criterion':['gini', 'entropy'],
        'max_samples':[None,0.5,0.75,1]
    }},

    # Logistic Regression model with a maximum iteration limit and parallel jobs
    'Logistic_Regression': {'model': LogisticRegression(max_iter=10000,random_state=42,n_jobs=nr_jobs),
    'params':{
             'penalty' : ['l1', 'l2', 'elasticnet','none'],
        'C' : np.logspace(-4, 4, 20),
        'fit_intercept':[True,False],
        'class_weight':[None,'balanced'],
        'solver': ['lbfgs','newton-cg','liblinear','sag','saga','newton-cholesky']
    }},
    
    # Gaussian Naive Bayes model
    'Gaussian_Naive_Bayes': {
        'model': GaussianNB(),
        'params':{
            'var_smoothing': [10**-i for i in range(2, 16)]
    }},
    
    # Support Vector Classifier with specified random state
    'SVM': {
        'model': SVC(random_state=42),
        'params':{
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto',1, 0.1, 0.01, 0.001, 0.0001],
                'class_weight': [None, 'balanced'],
                'probability':[True]
        }
      }
    }
    
    # Check if the provided classifier name exists in the dictionary
    if classifier_name not in classifiers:
        raise ValueError(f"Classifier '{classifier_name}' not found in the dictionary.")
    
    # Return the specified classifier's model and hyperparameter grid
    return classifiers[classifier_name]
    
