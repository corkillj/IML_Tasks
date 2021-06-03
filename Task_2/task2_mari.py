#%% 

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

#%%

X_train = pd.read_csv('train_features.csv')
y_train = pd.read_csv('train_labels.csv', index_col='pid')

#%%
X_test = pd.read_csv('test_features.csv')
pids_test = X_test['pid'].unique()

#%%
task1_feats = ['Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb',
       'HCO3', 'BaseExcess', 'Fibrinogen', 'Phosphate', 'WBC',
       'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
       'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
       'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
task1_labels_org = ['BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos', 'Bilirubin_total', 'Lactate', 'TroponinI', 'SaO2',
                'Bilirubin_direct', 'EtCO2', 'Sepsis']
task1_labels = []
for feat in task1_labels_org:
    task1_labels.append('LABEL_' + feat)
del task1_labels_org
    

task3_feats = ['Age', 'RRate', 'ABPm', 'SpO2', 'Heartrate']
task3_labels_org = ['RRate', 'ABPm', 'SpO2', 'Heartrate']
task3_labels = []
for feat in task3_labels_org:
    task3_labels.append('LABEL_' + feat)
del task3_labels_org

num_patients = X_train['pid'].nunique()
pids = X_train['pid'].unique()


#%%
def preprocess_task1(X, pids, feats):
    X_out = pd.DataFrame(index=pids)

    for feat in feats:
        print('Processing feature ' + feat)
        for pid in pids:
            if feat == 'Age':
                X_out.loc[pid, feat] = X[X['pid'] == pid][feat].iloc[0]
                continue
            
            X_out.loc[pid, 'max' + feat] = X[X['pid'] == pid][feat].max()
            X_out.loc[pid, 'min' + feat] =  X[X['pid'] == pid][feat].min()
            X_out.loc[pid, 'med' + feat] =  X[X['pid'] == pid][feat].median()
            X_out.loc[pid, 'num' + feat] = X[X['pid'] == pid][feat].count()
            X_out['num' + feat].fillna(0, inplace=True)
            
    return X_out

#%%
X_task1 = preprocess_task1(X_train, pids, task1_feats)
#%%
X_task1.to_csv('X_task1.csv', index_label='pid')

#%%
X_task1 = pd.read_csv('X_task1.csv', index_col='pid')

#%%
def preprocess_task3(X, pids, feats):
    X_out = pd.DataFrame(index=pids)
    
    for feat in task3_feats:
        print('Processing feature ' + feat)
        for pid in pids:
            if feat == 'Age':
                X_out.loc[pid, feat]= X[X['pid'] == pid][feat].iloc[0]
            
            # num_steps = len(X[X['pid'] == pid][feat])
            # if num_steps != 12:
            #     print('num_steps is ' + str(num_steps) + ' for pid ' + str(pid) + ' instead of 12!')
            # X_out.loc[pid, 'med1' + feat] = X[X['pid'] == pid][feat].iloc[:num_steps//2].median()
            X_out.loc[pid, 'med1' + feat] = X[X['pid'] == pid][feat].median()
            X_out.loc[pid, 'meanDiff' + feat] = X[X['pid'] == pid][feat].diff().mean()
            X_out.loc[pid, 'min' + feat] = X[X['pid'] == pid][feat].min()
            X_out.loc[pid, 'max' + feat] = X[X['pid'] == pid][feat].max()
            
    return X_out
#%%
X_task3 = preprocess_task3(X_train, pids, task3_feats)
#%%
X_task3.to_csv('X_task3.csv', index_label='pid')

#%%
X_task3 = pd.read_csv('X_task3.csv', index_col='pid')

#%%
from sklearn.preprocessing import StandardScaler
scalert1 = StandardScaler()
scalert1.fit(X_task1)
X_task1 = pd.DataFrame(scalert1.transform(X_task1), columns=X_task1.columns, index=X_task1.index)
#%%

from sklearn.model_selection import train_test_split
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_task1, y_train[task1_labels], test_size=0.2, random_state=1724)
X_tr1 = X_tr1.fillna(0)
X_te1 = X_te1.fillna(0)
#%%

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
y_pre1 = pd.DataFrame()
y_pre_proba1 = pd.DataFrame()
clf1 = [None]*len(task1_labels)
for i, label in enumerate(task1_labels):
    # print('Fitting ' + label)
    clf1[i] = SVC(C=0.2, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=1724,
              verbose=True, max_iter=50000, cache_size=4000)
    clf1[i].fit(X_tr1, y_tr1[label])
    y_pre1[label] = clf1[i].predict(X_te1)
    y_pre_proba1[label] = clf1[i].predict_proba(X_te1)[:, 1]
    print('BAAC: ' + str(balanced_accuracy_score(y_te1[label], y_pre1[label])))
    print('ROC_AUC (class): ' + str(roc_auc_score(y_te1[label], y_pre1[label])))
    print('ROC_AUC (proba): ' + str(roc_auc_score(y_te1[label], y_pre_proba1[label])))
    

#%%
# Task 3
from sklearn.preprocessing import StandardScaler
scalert3 = StandardScaler()
scalert3.fit(X_task3)
X_task3 = pd.DataFrame(scalert3.transform(X_task3), columns=X_task3.columns, index=X_task3.index)

#%%
from sklearn.model_selection import train_test_split
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X_task3, y_train[task3_labels], test_size=0.2, random_state=1724)
X_tr3 = X_tr3.fillna(X_tr3.mean()) 
X_te3 = X_te3.fillna(X_te3.mean())
#%%
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
y_pre3 = pd.DataFrame()
clf3 = [None]*len(task3_labels)
for i, label in enumerate(task3_labels):
    print('Fitting ' + label)
    clf3[i] = Ridge(alpha=0.1, random_state=1724)
    clf3[i].fit(X_tr3, y_tr3[label])
    y_pre3[label]  = clf3[i].predict(X_te3)
    print('R2: ' + str(r2_score(y_te3[label], y_pre3[label])))
    

#%%
# Submission
y_total = pd.DataFrame(index=pids_test)
#%%
# task1
X_test_t1 = preprocess_task1(X_test, pids_test, task1_feats)
X_test_t1.to_csv('X_test_t1.csv', index_label='pid')

#%%
X_test_t1 = pd.read_csv('X_test_t1.csv', index_col='pid')

X_test_t1 = pd.DataFrame(scalert1.transform(X_test_t1), columns=X_test_t1.columns, index=X_test_t1.index)
#%%
for i, label in enumerate(task1_labels):
    clf1[i].fit(X_task1.fillna(0), y_train[label])
    y_total[label] = clf1[i].predict_proba(X_test_t1.fillna(0))[:, 1]
    
#%%
# task 3
X_test_t3 = preprocess_task3(X_test, pids_test, task3_feats)
X_test_t3.to_csv('X_test_t3.csv', index_label='pid')

#%%
X_test_t3 = pd.read_csv('X_test_t3.csv', index_col='pid')

X_test_t3 = pd.DataFrame(scalert3.transform(X_test_t3), columns=X_test_t3.columns, index=X_test_t3.index)

#%%
for i, label in enumerate(task3_labels):
    clf3[i].fit(X_task3.fillna(X_task3.mean()), y_train[label])
    y_total[label] = clf3[i].predict(X_test_t3.fillna(X_test_t3.mean()))

#%%
y_total.to_csv('y_pred_mari2.zip', index_label='pid', float_format='%.3f', compression='zip')
