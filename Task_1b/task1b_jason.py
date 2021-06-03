# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:00:36 2021

@author: Jason
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

train_set_linear = np.genfromtxt('train.csv', delimiter=',')[1:,1:]
train_set_quadratic = train_set_linear**2
train_set_exponential = np.exp(train_set_linear)
train_set_cosine = np.cos(train_set_linear)


train_set_full = np.concatenate((train_set_linear, train_set_quadratic,\
                                 train_set_exponential, train_set_cosine, np.ones((np.shape(train_set_linear)[0],1))),axis=1)
    

'''Attempt with panda'''
pd_ts = pd.read_csv('train.csv',index_col=0)
y = pd_ts['y']
pd_ts = pd_ts.drop(columns=['y'])
pd_ts_q = pd_ts**2
pd_ts_exp = np.exp(pd_ts)
pd_ts_cos = np.cos(pd_ts)
pd_ones = pd.Series(np.ones((np.shape(pd_ts)[0])))
pd_ts_full = pd.concat([pd_ts, pd_ts_q, pd_ts_exp, pd_ts_cos, pd_ones  ], axis=1)

clf = LassoCV(alphas = np.logspace( -3,3, num=5000), fit_intercept=False, max_iter=10000)
w = clf.fit(pd_ts_full,y)
weights = w.coef_
np.savetxt('submission_Jason_LassoCV.csv', weights, delimiter = ',' )

#np.logspace(-2,3, num=40)