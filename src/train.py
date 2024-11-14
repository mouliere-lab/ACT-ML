# Author: Parisa Mapar

from sklearn.model_selection import GridSearchCV, ParameterGrid
import logging
import warnings

# Setting up logging configuration for displaying messages with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Silence warnings during execution
warnings.filterwarnings("ignore")

# This function trains a classifier and tunes its hyperparameters.
def train_classifier(train_set, features, target_train, classifier_info, classifier_name, nr_jobs,cv):

    try:
        # Extract the classifier from the classifier_info dictionary
        classifier = classifier_info['model']

        # Finding the optimal parameters through GridSearchCV and via the best accuracy
        optimized_classifier = GridSearchCV(classifier, param_grid=classifier_info['params'], scoring='accuracy', cv=cv, verbose=True, n_jobs=nr_jobs)
        optimized_classifier.fit(train_set[features], target_train)
        
        # Log the best parameters found by GridSearchCV
        logging.info(f"Best parameters: {optimized_classifier.best_params_}")
            
        return optimized_classifier
            
    except Exception as e:
    
        # Log any error that occurs during training and re-raise the exception for further handling
        logging.error(f"Error in training classifier: {e}")
        raise

