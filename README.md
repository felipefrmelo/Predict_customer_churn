# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The move from a Data Scientist to a Machine Learning Engineer requires a move to coding best practices. In this project, we are tasked with moving code from a notebook that completes the data science process, but doesn't lend itself easy to reproduce, production-level code, to two scripts:

1. The first script `churn_library.py` is a python library containing functions needed to complete the same data science process.

2. The second script `churn_script_logging_and_tests.py` contains tests and logging that test the functions of your library and log any errors that occur.  

The original python notebook `churn_notebook.ipynb` contains the code to be refactored.

The new code was formatted using [autopep8](https://pypi.org/project/autopep8/), and both scripts provided [pylint](https://pypi.org/project/pylint/) scores exceeding **8.8**.

## Running Files
All python libraries used in this repository can be `pip` installed.  All files were created and tested using Python **3.9**.  


If you have the same folder structure as this repository, as well as the data available from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers?select=BankChurners.csv), then you can run the commands below to retrieve all results.

This following will test each of the functions and provide any errors to a file stored in the `logs` folder.
```
ipython churn_script_logging_and_tests.py
```
This following will contain all the functions and refactored code associated with the original notebook.
```
ipython churn_library.py
```

You can also check the pylint score, as well as perform the auto-formatting using the following commands:

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

The files here were formated using:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```
