# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 09:32:30 2021

@author: Adrian
"""

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

train = np.genfromtxt('train.csv',delimiter=',',skip_header=1)

samples = train[:,2:]
y = train[:,1].reshape(-1,1)

X = np.concatenate((samples,samples**2,np.exp(samples),np.cos(samples),np.ones((samples.shape[0],1))),axis=1)


w = np.linalg.inv(X.T@X)@X.T@y

print("Regression Weights: ",w.T)

reg = linear_model.Ridge(alpha=0.001,fit_intercept=False)

reg.fit(X,y)

print("SK Coeff: ",reg.coef_)
coef = reg.coef_
print("SK intercept: ",reg.intercept_)

y_pred= X@coef.T
rsme= mean_squared_error(y_pred,y)**0.5

print("RSME: ", rsme)



np.savetxt('result_ridge.csv',coef.T,fmt='%f',header="",comments="")