"""
author: Felipe Melo
date: September 30, 2021
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import seaborn as sns
sns.set()


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


def save_figure(data, fig_name, strategy, **kwargs):
    '''
    saves figure to images folder
    input:
            data: data to plot
            fig_name: name of figure
            strategy: plotting strategy
            kwargs: arguments for plotting strategy
    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    strategy(data=data, **kwargs)
    plt.savefig(f"images/eda/{fig_name}")
    plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    save_figure(df["Churn"], "churn_distribution", sns.histplot)

    save_figure(df["Customer_Age"], "age_distribution", sns.histplot)

    save_figure(df, "marital_distribution", sns.catplot,
                kind="count", x="Marital_Status")

    save_figure(
        df["Total_Trans_Ct"],
        "total_transaction_distribution",
        sns.histplot,
        kde=True)

    save_figure(df.corr(), "heatmap", sns.heatmap,
                annot=False, cmap='Dark2_r', linewidths=2)


def encoder_helper(df, category_lst, response, keep_cols=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for
    '''
    new_df = df.copy()
    for col in category_lst:

        cat_lst = []
        cat_groups = df.groupby(col).mean()[response]

        for val in df[col]:
            cat_lst.append(cat_groups.loc[val])

        new_df[f"{col}_{response}"] = cat_lst

    if keep_cols is not None:
        new_df = new_df[keep_cols]
    return new_df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name  index y column

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df[response]

    keep_cols = [
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
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    X = encoder_helper(df, category_lst, response, keep_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def save_report(y_train, y_test, y_train_preds, y_test_preds, fig_name):
    '''
    save classification report to images folder
    input:
            y_train: y training data
            y_test: y testing data
            y_train_preds: y training predictions
            y_test_preds: y testing predictions
            fig_name: name of figure
    output:
            None
    '''

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(f'{fig_name.capitalize()} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.06, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(f'{fig_name.capitalize()} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        f"images/results/{fig_name.lower().replace(' ', '_')}_classification_report.png")
    plt.close()


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
    save_report(y_train, y_test, y_train_preds_rf,
                y_test_preds_rf, "Random Forest")

    save_report(y_train, y_test, y_train_preds_lr,
                y_test_preds_lr, "Logistic Regression")


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)

    plt.close()


def roc_curve_helper(y_test, x_test, models):
    '''
    creates and stores the ROC curve in pth
    input:
            y_test: test response values
            x_test: test X values
            models: list of models
            fig_name: name of figure to be saved

    output:
             None
    '''

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    for model in models:
        plot_roc_curve(model, x_test, y_test, ax=ax, alpha=0.8)
    plt.show()
    plt.savefig(
        "images/results/roc_curve.png")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    roc_curve_helper(y_test, x_test, [cv_rfc.best_estimator_, lrc])

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # feature importance
    feature_importance_plot(cv_rfc.best_estimator_, x_train,
                            './images/results/feature_importance_rf.png')

    # classification report
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
