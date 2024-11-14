# Author: Parisa Mapar

import logging

# This function tests a trained classifier on the provided test set.
def test_classifier(classifier, test_set, features):

    try:
        # Predict the class labels for the test set
        y_test_pred = classifier.predict(test_set[features])
        
        # Predict the probability of the positive class for the test set
        y_test_proba = classifier.predict_proba(test_set[features])[:, 1]
        
        return y_test_pred, y_test_proba
        
    except Exception as e:
        # Log any errors encountered during prediction
        logging.error(f"Error in testing classifier: {e}")
        raise
        
