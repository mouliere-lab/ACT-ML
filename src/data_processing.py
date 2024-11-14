# Author: Parisa Mapar

import logging

# This function preprocesses the input data by sorting it and filtering for specific response values.
def preprocess_data(data):
    try:
        # Sort the data by the 'LP' column and reset the index, dropping the old index.
        data = data.sort_values(by=['LP']).reset_index(drop=True)
        
        # Filter the data to include only rows where 'EOT_response' is 'Responder' or 'Non_Responder'.
        data = data[(data['EOT_response'] == 'Responder') | (data['EOT_response'] == 'Non_Responder')]
        
        # Return the processed data.
        return data
        
    except KeyError as e:
        # Log an error if a required column is missing and re-raise the exception.
        logging.error(f"Required columns not found in the data: {e}")
        raise
