#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:51:13 2021

@author: mariankannwischer
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%%
X_train = pd.read_csv('train.csv')

#%%
y_train = X_train['Active']

#%%
# Convert Strings to bytes
def preprocess_data(X):
    byte_seqs = [list(sample['Sequence'].encode()) for i, sample in X.iterrows()]
    
    X_proc = pd.DataFrame(byte_seqs, columns=['a1', 'a2', 'a3', 'a4'])
    return X_proc

#%%
X_proc = preprocess_data(X_train)
#%%
X_tr, X_te, y_tr, y_te = train_test_split(X_proc, y_train, random_state=1724)

#%%
import xgboost as xgb
clf = xgb.XGBClassifier()
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)

#%%
from sklearn.metrics import f1_score
print(f1_score(y_te, y_pred))

#%%
#Submission
clf.fit(X_proc, y_train)

#%%
X_test = pd.read_csv('test.csv')

#%%
X_test_proc = preprocess_data(X_test)

#%%
y_pred_all = clf.predict(X_test_proc)

#%%
y_sub = pd.read_csv('sample.csv', names=['Sequence'])
#%%
y_sub['Sequence'] = y_pred_all

#%%
y_sub.to_csv('pred_mari_xgb1.csv', index=False, header=False)