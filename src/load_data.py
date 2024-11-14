# Author: Parisa Mapar

import pandas as pd
import logging

# This function loads data from an Excel file and handles potential errors during the process.
def load_data(file_path):
    try:
        # Attempt to read the Excel file into a pandas DataFrame.
        data = pd.read_excel(file_path)
        
        # Return the loaded data if successful.
        return data
        
    except Exception as e:
    
        # Log an error if there is an issue reading the file and re-raise the exception.
        logging.error(f"Error in reading the input file: {e}")
        raise
    


