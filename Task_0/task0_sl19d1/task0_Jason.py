# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:10:11 2021

@author: Jason
"""

from sklearn.metrics import mean_squared_error
import numpy as np

train_set = np.genfromtxt('train.csv', delimiter=',')

X = np.concatenate((train_set[1:,2:], np.ones((np.shape(train_set)[0]-1,1))), axis=1)
y = train_set[1:,1]

def linear_regression_closedform(X,y):
    inv = np.linalg.inv(X.T@X)@X.T
    w = inv@y
    print("Weights: " , w)
    return(w)

w_est = linear_regression_closedform(X, y)
y_pred = X @ w_est

RMSE = mean_squared_error(y, y_pred)**0.5
print(RMSE)

test_set = np.genfromtxt('test.csv', delimiter=',')
X_test = np.concatenate((test_set[1:,1:], np.ones((np.shape(test_set)[0]-1,1))), axis=1)
y_test_pred = X_test @ w_est
ids = test_set[1:,0]
submission = np.concatenate((ids.reshape(-1,1),y_test_pred.reshape(-1,1)), axis = 1)
np.savetxt('submission.csv', submission, delimiter=',', header= "Id,y", comments='')