import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load iris dataset as an example
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# train a k-nearest neighbors classifier on the training data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)

# make predictions on the dataset
Y_pred = knn.predict(X_test)

# evaluate models accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


