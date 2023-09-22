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


def perform_eda(df, pth="./images/eda/"):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            pth: path for saving EDA files

    output:
            None
    '''
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio'
    ]
    df['Churn'] =\
        df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    image_path = pth + "churn_histogram.png" # "./images/eda/churn_histogram.png"
    plt.savefig(image_path)
    plt.close()

    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    image_path = pth + "customer_age_histogram.png"
    plt.savefig(image_path)
    plt.close()

    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar');
    image_path = pth + "marital_status_value_counts_bar.png"
    plt.savefig(image_path)
    plt.close()

    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True);
    image_path = pth + "total_trans_ct_histplot.png"
    plt.savefig(image_path)
    plt.close()

    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.show()
    image_path = pth + "feature_corr_heatmap.png"
    plt.savefig(image_path)
    plt.close()
    

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
    for category in category_lst:
        category_lst = []
        category_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            category_lst.append(category_groups.loc[val])

        df[f'{category}_{response}'] = category_lst

    return df


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

    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    response = 'Churn'

    post_encoding_df = encoder_helper(df, category_lst, response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = post_encoding_df[keep_cols]
    y = df['Churn']
    
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


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

if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)

