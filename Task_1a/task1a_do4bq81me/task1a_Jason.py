# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 14:27:55 2021

@author: Jason
"""

import numpy as np

lambdas = np.array([0.1,1,10,100,200])
k=10
train_set = np.genfromtxt('train.csv', delimiter=',')[1:,:]


len_train = np.size(train_set, axis=0)
len_set = int(len_train/k)

CV_all = np.zeros(len(lambdas))
#%%
def ridge_reg(X,y,l):
    inv = np.linalg.inv(X.T @ X + l* np.eye(np.size(X, axis=1) )) @ X.T
    w = inv @ y
    return w
#%%
j=0
for l in lambdas:

    '''split train data into k equal bits'''
    R_D_r_i = 0
    for i in range(k):
        '''extract Di'''
        X_i = np.concatenate((train_set[i*len_set:(i+1)*len_set,1:], np.ones((len_set,1))), axis=1)
        y_i = train_set[i*len_set: (i+1)*len_set, 0]


        '''caluculate R_D'i   ( D' = D_r )'''
        X_r = np.delete( np.concatenate( (train_set[:,1:] , np.ones((len_train ,1))), axis =1),\
                        slice(i*len_set,(i+1)*len_set ), axis=0)
        y_r = np.delete(train_set[:,0], slice(i*len_set,(i+1)*len_set ), axis=0)

        '''do ridge regression'''
        w = ridge_reg(X_r,y_r,l)

        R_D_r_i = R_D_r_i + (np.sum((y_i - X_i @ w)**2)/len_set)**0.5


    CV_all[j] = R_D_r_i / k
    j = j+1


np.savetxt('submission_jason.csv', CV_all, delimiter=',')
