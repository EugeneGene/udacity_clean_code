"""
This Python library provides a set of functions for performing binary classification on a dataset
to predict churn. It includes data import, exploratory data analysis (EDA), feature engineering,
model training and evaluation, as well as visualization functions.

Dependencies:
- os
- shap
- joblib
- pandas
- numpy
- matplotlib
- seaborn
- sklearn

Functions:
- `import_data(pth)`:
    Loads a CSV file located at 'pth' into a pandas DataFrame.
- `perform_eda(df)`:
    Performs exploratory data analysis on the input DataFrame and saves figures
    to an 'images' folder.
- `encoder_helper(df, category_lst, response)`:
    Converts categorical columns into new columns with the proportion of churn
    for each category.
- `perform_feature_engineering(df, response)`:
    Prepares data for model training and returns training and testing datasets.
- `classification_report_image(y_train, y_test, y_train_preds_lr,
                               y_train_preds_rf, y_test_preds_lr,
                               y_test_preds_rf)`:
    Produces classification reports for training and testing results
    and stores them as images.
- `feature_importance_plot(model, X_data, output_pth)`:
    Creates and stores feature importance plots.
- `train_models(X_train, X_test, y_train, y_test)`:
    Trains classification models, stores model results (images and scores),
    and stores the trained models.

Usage:
To use this library, import it in your Python script and call the provided functions as needed.

For example:
```python
import churn_library as cls

# Load data
data = cls.import_data("./data/bank_data.csv")

# Perform EDA
cls.perform_eda(data)

# Encode categorical features
encoded_data = cls.encoder_helper(data, category_lst=['category1', 'category2'], response='churn')

# Prepare data for modeling
X_train, X_test, y_train, y_test = cls.perform_feature_engineering(encoded_data, response='churn')

# Train models and store results
cls.train_models(X_train, X_test, y_train, y_test)

Please make sure to provide the necessary data and adjust the function parameters as needed.
"""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
                      could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass
