# Data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Models
from logistic_regression import MyLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# FaNcY MaTh
import numpy as np

# Remove anoying warnings
from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning)


# Data
data = load_breast_cancer()
X, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Accuracy calculation function
def accuracy(y_test, y_pred):
    return np.sum(y_pred == y_test)/len(y_test)


# Models
model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f'Sklearn Logistic Regression accuracy: {round((accuracy(y_test, y_pred) * 100), 1)}%')

my_model = MyLogisticRegression()
my_model.fit(x_train, y_train)
y_pred = my_model.predict(x_test)
print(f'My Logistic Regression accuracy: {round((accuracy(y_test, y_pred) * 100), 1)}%')

svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(f'SVC linear accuracy: {round((accuracy(y_test, y_pred) * 100), 1)}%')

kneighbors = KNeighborsClassifier(n_neighbors=10)
kneighbors.fit(x_train, y_train)
y_pred = kneighbors.predict(x_test)
print(f'KNeighbors accuracy: {round((accuracy(y_test, y_pred) * 100), 1)}%')