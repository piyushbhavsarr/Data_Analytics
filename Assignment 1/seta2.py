import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create 'realestate' data set with random data
np.random.seed(42)
n = 500
ids = np.arange(1, n+1)
flats = np.random.randint(1, 5, n)
houses = np.random.randint(1, 6, n)
purchases = 10_000 + flats * 5_000 + houses * 10_000 + np.random.normal(0, 5_000, n)
realestate_df = pd.DataFrame({'ID': ids, 'flat': flats, 'houses': houses, 'purchases': purchases})

# Split data into independent and target variables
X = realestate_df[['flat', 'houses']]
y = realestate_df['purchases']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print training and testing sets
print("Training set:")
print(X_train.head())
print(y_train.head())
print("\nTesting set:")
print(X_test.head())
print(y_test.head())

# Build linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Linear regression model score:", lr_model.score(X_test, y_test))

# Build simple linear regression model for predicting purchases
slr_model = LinearRegression()
slr_model.fit(X_train[['flat']], y_train)
y_pred_slr = slr_model.predict(X_test[['flat']])
print("Simple linear regression model score:", slr_model.score(X_test[['flat']], y_test))
