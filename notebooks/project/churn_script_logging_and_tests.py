"""
This test script is designed to test various functions from the 'churn_library'
 module, which provides functionality for churn prediction tasks.
 It includes test cases for data import, exploratory data analysis (EDA),
 feature encoding, feature engineering, model training, and evaluation.

Dependencies:
- os
- logging
- churn_library

Test Functions:
- `test_import(import_data)`:
    Tests the data import functionality by loading a sample CSV file. It checks
    if the DataFrame has non-zero dimensions after import.
- `test_eda(perform_eda)`:
    Placeholder for testing the 'perform_eda' function for exploratory data
    analysis.
- `test_encoder_helper(encoder_helper)`:
    Placeholder for testing the 'encoder_helper' function for encoding
    categorical features.
- `test_perform_feature_engineering(perform_feature_engineering)`:
    Placeholder for testing the 'perform_feature_engineering' function for
    feature engineering.
- `test_train_models(train_models)`: Placeholder for testing the 'train_models'
    function for model training and evaluation.
"""
import os, shutil
import logging
import churn_library as cls
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: The file wasn't found")
        raise err
        
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    # Create a sample DataFrame for testing
    sample_data = {
        'index': [1, 2, 3, 4, 5, 6],
        'Attrition_Flag': ["Existing Customer", "Existing Customer",
                           "Attrited Customer", "Attrited Customer", 
                           "Existing Customer", "Existing Customer"],
        'Customer_Age': [67, 35, 20, 18, 66, 68],
        'Marital_Status': ["Unknown", "Divorced",
                          "Divorced", "Married", 
                          "Divorced", "Divorced"],
        'Total_Trans_Ct': [42, 33, 20, 20, 28, 24]
    }
    sample_data_df = pd.DataFrame(sample_data)

    # Specify a test directory for images (you may need to adjust this path)
    test_images_directory = "./images_test/eda/"

    # Create the test directory if it doesn't exist
    if not os.path.exists(test_images_directory):
        os.makedirs(test_images_directory)

    try:
        # Call perform_eda on the sample DataFrame
        perform_eda(sample_data_df, test_images_directory)

        # Get a list of all files and directories in the specified directory
        files_and_directories = os.listdir(test_images_directory)
        # Use a list comprehension to filter only files (not directories)
        files = [file for file in files_and_directories if 
                 os.path.isfile(os.path.join(test_images_directory, file))]

        # Check if the images were generated in the test directory
        assert len(files) > 0
        logging.info(f"SUCCESS: {len(files)} EDA File Created.")
        
    except:
        logging.error("Error: EDA file(s) weren't created")
        
    # Comment finally to see test plots
    finally:
        # Clean up: remove generated images and the test directory
        for file in os.listdir(test_images_directory):
            file_path = os.path.join(test_images_directory, file)
            os.remove(file_path)
            logging.info(f"SUCCESS: {file_path} removed")
        shutil.rmtree(test_images_directory)
        logging.info(f"SUCCESS: {test_images_directory} removed")

    

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''

    test_data_df = cls.import_data("./data_test/bank_data_test.csv")
    test_data_df['Churn'] =\
    test_data_df['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)

    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    response = 'Churn'

    
    post_encoding_df = encoder_helper(test_data_df, category_lst, response)

    # Define the value you want to check for
    new_cols_to_check = ['Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn']

    
    for new_col_to_check in new_cols_to_check:
        # Test for the existence of the name in the 'Name' column
        try:
            assert new_col_to_check in post_encoding_df.columns
            logging.info(f"SUCCESS: {new_col_to_check} encoding completed.")
    
        except AssertionError as e:
            logging.error(f"Error: {e}")
            

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    test_data_df = cls.import_data("./data_test/bank_data_test.csv")
    test_data_df['Churn'] =\
    test_data_df['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)

    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    response = 'Churn'
    
    post_encoding_test_data_df = cls.encoder_helper(test_data_df, category_lst, response)

    X_train, X_test, y_train, y_test =\
        perform_feature_engineering(post_encoding_test_data_df, response)

    # Test for existence of training and testing set
    try:
        assert len(X_train) > 0
        logging.info(f"SUCCESS: X_train has a lenght of {len(X_train)}.")

    except AssertionError as e:
        logging.error(f"Error: {e}")

    try:
        assert len(X_test) > 0
        logging.info(f"SUCCESS: X_test has a lenght of {len(X_test)}.")

    except AssertionError as e:
        logging.error(f"Error: {e}")

    try:
        assert len(y_train) > 0
        logging.info(f"SUCCESS: y_train has a lenght of {len(y_train)}.")

    except AssertionError as e:
        logging.error(f"Error: {e}")

    try:
        assert len(y_test) > 0
        logging.info(f"SUCCESS: y_train has a lenght of {len(y_test)}.")

    except AssertionError as e:
        logging.error(f"Error: {e}")


def test_train_models(train_models):
    '''
    test train_models
    '''
    test_data_df = cls.import_data("./data_test/bank_data_test.csv")
    test_data_df['Churn'] =\
    test_data_df['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)

    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    response = 'Churn'
    
    post_encoding_test_data_df = cls.encoder_helper(test_data_df, category_lst, response)

    X_train, X_test, y_train, y_test =\
        cls.perform_feature_engineering(post_encoding_test_data_df, response)



    
    
    # Test

    # Specify a test directory for images (you may need to adjust this path)
    test_results_directory = "./images_test/results/"
    test_models_directory = "./models_test/"

    # Create the directories if they don't exist
    for dir in [test_results_directory, test_models_directory]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    train_models(X_train, X_test, y_train, y_test, 
                 test_results_directory, 
                 test_models_directory)

    

    
    # finally:
    #     # Clean up: remove generated test results and models
    #     for dir in [test_results_directory, test_models_directory]:
    #         for file in os.listdir(dir):
    #             file_path = os.path.join(dir, file)
    #             os.remove(file_path)
    #             logging.info(f"SUCCESS: {file_path} removed")
    #         os.rmdir(dir)
    #         logging.info(f"SUCCESS: {dir} removed")



if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
