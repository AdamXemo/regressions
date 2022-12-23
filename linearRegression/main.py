from linear_regression import MyLinearRegression
from json import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, QuantileRegressor, SGDRegressor
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBRFRegressor


# Data

data = load(open('data.txt', 'r'))
X = data[0]
y = data[1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
x_train = np.array(x_train)
x_test = np.array(x_test)


# Models

my_linear = MyLinearRegression()
my_linear.fit(x_train, y_train)
y_pred = my_linear.predict(x_test)

x_train.resize(len(x_train), 1)
x_test.resize(len(x_test), 1)

sklearn_linear = LinearRegression()
sklearn_linear.fit(x_train, y_train)
y_pred2 = sklearn_linear.predict(x_test)

sklearn_kernel_ridge = KernelRidge()
sklearn_kernel_ridge.fit(x_train, y_train)
y_pred3 = sklearn_kernel_ridge.predict(x_test)

xgbr = XGBRegressor()
xgbr.fit(x_train, y_train)
y_pred4 = xgbr.predict(x_test)

xgbrf = XGBRFRegressor()
xgbrf.fit(x_train, y_train)
y_pred5 = xgbrf.predict(x_test)

svr = svm.SVR(kernel='poly')
svr.fit(x_train, y_train)
y_pred6 = svr.predict(x_test)

lasso = Lasso()
lasso.fit(x_train, y_train)
y_pred7 = lasso.predict(x_test)

quantile = QuantileRegressor()
quantile.fit(x_train, y_train)
y_pred8 = quantile.predict(x_test)

sgd = SGDRegressor()
sgd.fit(x_train, y_train)
y_pred9 = sgd.predict(x_test)


# Graphs

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred)
plt.title('My Linear Regression')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred2)
plt.title('Sklearn Linear Regression')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred3)
plt.title('Kernel Ridge')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred6)
plt.title('Support Vector Regression (poly)')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred7)
plt.title('Lasso Regression')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred9)
plt.title('SGDRegressor')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred8)
plt.title('Quantile Regressor')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred4)
plt.title('XGBRegressor')
plt.show()

plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred5)
plt.title('XGBRFRegressor')
plt.show()