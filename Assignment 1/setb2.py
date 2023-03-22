import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Print out some basic statistical details for each species
print("Iris-setosa:")
print(iris[iris.species=='Iris-setosa'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())
print("\nIris-versicolor:")
print(iris[iris.species=='Iris-versicolor'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())
print("\nIris-virginica:")
print(iris[iris.species=='Iris-virginica'][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].describe())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']], iris['species'], test_size=0.2, random_state=0)

# Create a logistic regression model and fit it to the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Use the trained model to predict the species of the flowers in the testing set
y_pred = lr.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
