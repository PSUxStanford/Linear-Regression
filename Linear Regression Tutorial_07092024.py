# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:05:51 2024

@author: Samuel Stanford
"""

"Basic Script for a Linear Regression Model"
import matplotlib.pyplot as plt
from sklearn import datasets

diabetes = datasets.load_diabetes()

#print(diabetes.DESCR)

"Data is coming from the feature names,"
"and the target Y value is the disease progression"
X1 = diabetes.data
Y1 = diabetes.target

#print(X.shape, Y.shape)

from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, test_size = 0.2)

#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()

model.fit(X1_train, Y1_train)

Y1_pred = model.predict(X1_test)
print("Diabetes Linear Regression")
print('Coefficients:', model.coef_)
print(diabetes.feature_names)

print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y1_test, Y1_pred))
"%.2f means we will use 2 decimal places"
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y1_test, Y1_pred))

print(r2_score(Y1_test,Y1_pred).dtype)
print("_______________")


plt.figure(1)
plt.scatter(Y1_test, Y1_pred, marker="*", alpha = 0.25)
plt.title("Diabetes Data")


"Second Linear Regression, Boston Housing"

import pandas as pd
BostonHousing = pd.read_csv("https://github.com/dataprofessor/data/raw/master/BostonHousing.csv")
#print(BostonHousing)

Y2 = BostonHousing.medv
X2 = BostonHousing.drop(['medv'],axis=1)

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2,Y2, test_size = 0.2)
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


model = linear_model.LinearRegression()

model.fit(X2_train, Y2_train)

Y2_pred = model.predict(X2_test)

print("Boston Housing Linear Regression")
print('Coefficients:', model.coef_)

print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y2_test, Y2_pred))
"%.2f means we will use 2 decimal places"
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y2_test, Y2_pred))

print(r2_score(Y2_test,Y2_pred).dtype)

plt.figure(2)
plt.scatter(Y2_test, Y2_pred, color="red",marker="+", alpha = 0.25)
plt.title("Boston Housing Data")
