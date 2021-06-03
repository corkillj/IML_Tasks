# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:45:54 2021

@author: Jason
"""

import pandas as pd
import numpy as np
from sklearn import svm

hours_recorded = 12
train_features = pd.read_csv('train_features.csv')
train_labels = pd.read_csv('train_labels.csv')
test_features = pd.read_csv('test_features.csv')

''' 
Subtask 1
'''
tests = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', \
         'Lactate', 'TroponinI', 'SaO2', 'Bilirubin_direct', 'EtCO2']
t1_gt = train_labels.loc[:,'LABEL_BaseExcess':'LABEL_EtCO2']

prediction = pd.DataFrame(index = test_features['pid'].unique())
for test in tests:
    cur_test_gt = t1_gt['LABEL_'+test]
    cur_test_data = pd.DataFrame(index = train_labels['pid'])
    for pid in train_labels['pid']:
        cur_test_data.loc[pid, 'amount'+ test] = train_features[train_features['pid'] == pid][test].count() 
        
   
    clf = svm.SVC(probability = True, class_weight = 'balanced', C = 0.2, cache_size =2000 )
    clf.fit(cur_test_data, cur_test_gt)
    
    cur_test_test_data = pd.DataFrame(index = test_features['pid'].unique())
    for pid in test_features['pid'].unique():
        cur_test_test_data.loc[pid, 'amount'+ test] = test_features[test_features['pid'] == pid][test].count()
    
    prediction['LABEL_'test] = clf.predict_proba(cur_test_test_data)
    
    print(test)
    
'''
Subtask 2
'''    

    