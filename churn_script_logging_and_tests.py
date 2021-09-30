"""
Contains tests and logging that test the functions of your library and log any errors that occur.
author: Felipe Melo
date: September 30, 2021
"""

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
        files = os.listdir("./images/eda")
        assert len(files) > 0
    except AssertionError as err:
        logging.error("Testing perform_eda: It didn't create the images")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    try:
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        response = "Churn"

        df_encoded = encoder_helper(df, category_lst, response)
        collumns = [f"{col}_{response}" for col in category_lst]

        for col in collumns:
            assert col in df_encoded.columns

        logging.info("Testing encoder_helper: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The encoder didn't create the new columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
          x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, "Churn")
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The feature engineering didn't")
        logging.error(" create the correct data")
        raise err
    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(x_train, x_test, y_train, y_test)
    path = "./images/results/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Results image files not found")
        raise err

    path = "./models/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err


if __name__ == "__main__":
    DF = test_import(cl.import_data)

    test_eda(cl.perform_eda, DF)
    test_encoder_helper(cl.encoder_helper, DF)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cl.perform_feature_engineering, DF)

    test_train_models(cl.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
