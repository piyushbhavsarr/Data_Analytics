# Importing the necessary libraries/packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

%matplotlib inline

# Reading the sales.csv file and previewing the dataset
df = pd.read_csv('sales.csv')
print(df.head())

# Selecting the independent variable(s) and target variable
X = df[['TV']] # independent variable
y = df['Sales'] # target variable

# Splitting the dataset into training and testing sets with a 7:3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Training Set - X_train shape: ", X_train.shape, " y_train shape: ", y_train.shape)
print("Testing Set - X_test shape: ", X_test.shape, " y_test shape: ", y_test.shape)

# Building a simple linear regression model using the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluating the model using the testing set
y_pred = regressor.predict(X_test)

print("R-squared: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
