import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('real_estate_dataset.csv')

# Split the dataset into input variables (X) and target variable (y)
X = df[['house_age', 'distance_to_mrt']]
y = df['house_price']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the LinearRegression class
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Use the trained model to make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Print the mean squared error
print('Mean squared error: ', mse)
