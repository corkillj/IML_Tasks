# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 13:57:23 2021

@author: Adrian
"""

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def ridge_closedform(X,y,lam):
    #closed form solution
    w_cl = np.linalg.inv(X.T@X+lam*np.identity(np.min(X.shape)))@X.T@y    
    return w_cl


train_full = np.genfromtxt('train.csv',delimiter=',',skip_header=1)



lambs = [0.1, 1, 10, 100, 200]

kf = KFold(n_splits=10)
avg_errors = np.zeros(5)

for i,lam in enumerate(lambs):
    errors_RSME = np.zeros(10)
    j=0
    
    #train 10 times for cross validation
    for train,test in kf.split(train_full):
        #print("test array: ", test[0], " to ", test[-1])
        #extract trainingsdata
        X = np.concatenate((np.ones(train.shape[0]).reshape(-1,1),train_full[train,1:]),axis=1)
        y = train_full[train,0].reshape(-1,1)
        
        #train with Ridge
        w = ridge_closedform(X,y,lam)

        #evaluate
        y_pred = train_full[test,1:]@w[1:]+w[0]
        
        #get RSME
        errors_RSME[j] = mean_squared_error(train_full[test,0],y_pred)**0.5
        
        j += 1
        
    #print("Errors RSME: ", errors_RSME)
    avg_errors[i] = np.average(errors_RSME)
        
        
print("Average RSME Errors: ", avg_errors)

#write to file
np.savetxt('result_1a.csv',avg_errors,fmt='%f',header="",comments="")

