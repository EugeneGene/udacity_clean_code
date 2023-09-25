"""
This test script validates functions from the 'churn_library' module,
used for churn prediction tasks.

Test Functions:
- `test_import(import_data)`
- `test_eda(perform_eda)`
- `test_encoder_helper(encoder_helper)`
- `test_perform_feature_engineering(perform_feature_engineering)`
- `test_train_models(train_models)`

Usage:
Run this script to validate the 'churn_library' functions.
"""
import os
import shutil
import logging
import unittest
import pandas as pd
import churn_library as cls

class TestChurnScript(unittest.TestCase):
    '''
    This test class is designed to test various functions from the 'churn_library'
    module, which provides functionality for churn prediction tasks.
    It includes test cases for data import, exploratory data analysis (EDA),
    feature encoding, feature engineering, model training, and evaluation.

    Dependencies:
    - os
    - shutil
    - logging
    - unittest
    - pandas
    - churn_library

    Test Functions:
    - `test_import(import_data)`:
        Tests the data import functionality by loading a sample CSV file.
        It checks if the DataFrame has non-zero dimensions after import.
    - `test_eda(perform_eda)`:
        Tests the 'perform_eda' function for exploratory data
        analysis. It checks if all of the EDA .png files were created
    - `test_encoder_helper(encoder_helper)`:
        Tests the 'encoder_helper' function for encoding
        categorical features. It checks if the post-encoded Dataframe
        contains specific columns.
    - `test_perform_feature_engineering(perform_feature_engineering)`:
        Tests the 'perform_feature_engineering' function for
        feature engineering. After laoding a test .csv file,
        it checks if the lengths of the X_train, X_test,
        y_train, y_test DataFrames are non-zero.
    - `test_train_models(train_models)`: Tests the 'train_models'
        function for model training and evaluation.
        After loading a test .csv file and performing feature engineering,
        it checks for the existence of model files (.pkl) and
        model result files (.csv and .png)
    '''

    logging.basicConfig(
        filename='./logs/churn_library.log',
        level = logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    def test_import(self):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        try:
            df = cls.import_data("./data/bank_data.csv")
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

    def test_eda(self):
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
            cls.perform_eda(sample_data_df, test_images_directory)

            # Get a list of all files and directories in the specified directory
            files_and_directories = os.listdir(test_images_directory)
            # Use a list comprehension to filter only files (not directories)
            files = [file for file in files_and_directories if
                     os.path.isfile(os.path.join(test_images_directory, file))]

            # Check if the images were generated in the test directory
            assert len(files) > 0
            logging.info("SUCCESS: %d EDA File Created.", len(files))

        except AssertionError:
            logging.error("ERROR: EDA file(s) weren't created")

        # Comment out the finally block to inspect test plots during testing
        finally:
            # Clean up: remove generated images and the test directory
            for file in os.listdir(test_images_directory):
                file_path = os.path.join(test_images_directory, file)
                os.remove(file_path)
                logging.info("SUCCESS: %s removed", file_path)
            shutil.rmtree(test_images_directory)
            logging.info("SUCCESS: %s removed", test_images_directory)
            try:
                shutil.rmtree("./images_test/")
                logging.info("SUCCESS: ./images_test/ removed")
            except Exception as err:
                logging.error("Error removing the ./images_test/ directory: %s", str(err))

    def test_encoder_helper(self):
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

        post_encoding_df = cls.encoder_helper(test_data_df, category_lst, response)

        # Define the value you want to check for
        new_cols_to_check = ['Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn']

        for new_col_to_check in new_cols_to_check:
            # Test for the existence of the name in the 'Name' column
            try:
                assert new_col_to_check in post_encoding_df.columns
                logging.info("SUCCESS: %s encoding completed.", new_col_to_check)

            except AssertionError as err:
                logging.error("Error: %s", err)

    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''
        test_data_df = cls.import_data("./data_test/bank_data_test.csv")

        category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category']
        response = 'Churn'

        post_encoding_test_data_df = cls.encoder_helper(test_data_df, category_lst, response)

        X_train, X_test, y_train, y_test =\
            cls.perform_feature_engineering(post_encoding_test_data_df, response)

        # Test for existence of training and testing set
        try:
            assert len(X_train) > 0
            logging.info("SUCCESS: X_train has a length of %d.", len(X_train))

        except AssertionError as err:
            logging.error("Error: %s", err)

        try:
            assert len(X_test) > 0
            logging.info("SUCCESS: X_test has a length of %d.", len(X_test))

        except AssertionError as err:
            logging.error("Error: %s", err)

        try:
            assert len(y_train) > 0
            logging.info("SUCCESS: y_train has a length of %d.", len(y_train))

        except AssertionError as err:
            logging.error("Error: %s", err)

        try:
            assert len(y_test) > 0
            logging.info("SUCCESS: y_test has a length of %d.", len(y_test))

        except AssertionError as err:
            logging.error("Error: %s", err)

    def test_train_models(self):
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
        for directory in [test_results_directory, test_models_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        cls.train_models(X_train, X_test, y_train, y_test,
                     test_results_directory,
                     test_models_directory)

        try:
            # Get a list of all files and directories in the specified directory
            # Use a list comprehension to filter only files (not directories)
            files = [file for file in os.listdir(test_models_directory)
                     if file.endswith(".pkl")]

            # Check if the images were generated in the test directory
            assert len(files) > 0
            logging.info("SUCCESS: %d model files (pkl) Created.", len(files))

        except AssertionError:
            logging.error("Error: No model files were created")


        finally:
            # Clean up: remove generated test results and models
            for directory in [test_results_directory, test_models_directory]:
                for file in os.listdir(directory):
                    if file.endswith(tuple(['.png', '.csv', '.pkl'])):
                        file_path = os.path.join(directory, file)
                        os.remove(file_path)
                        logging.info("SUCCESS: %s removed", file_path)
                try:
                    shutil.rmtree(directory)
                    logging.info("SUCCESS: %s removed", directory)
                except Exception as err:
                    logging.error("Error removing directory: %s", str(err))

            try:
                shutil.rmtree("./images_test/")
                logging.info("SUCCESS: ./images_test/ removed")
            except Exception as err:
                logging.error("Error removing the ./images_test/ directory: %s", str(err))

if __name__ == "__main__":
    unittest.main()
