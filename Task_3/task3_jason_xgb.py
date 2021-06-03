# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:45:37 2021

@author: Jason
"""
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

#%%
'''Read in data'''
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train_raw = train_data['Sequence']
y_train = train_data['Active']
X_test_raw = test_data['Sequence']

#%%
''' Unused '''
def process_data2(X):
    X = np.array([ list(X[string]) for string in range(X.shape[0])])
    encoded_x = None
    for i in range(0, X.shape[1]):
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(X[:,i])
        feature = feature.reshape(X.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)

        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1)
    
    
    return encoded_x

#%%
'''Convert X_raw for XGB'''
def process_data_xgb(X_raw):
    X_char = pd.DataFrame([ list(X_raw[string]) for string in range(X_raw.shape[0])])
    ohe = OneHotEncoder()
    X = ohe.fit_transform(X_char).toarray()
    return X
#%%
'''Convert X_raw for CAT'''
def process_data_cat(X_raw):
    X_char = pd.DataFrame([ list(X_raw[string]) for string in range(X_raw.shape[0])])
    return X_char

#%%
''' Process the data '''
X_train_xgb = process_data_xgb(X_train_raw) #interestingly ther doesnt seem to be  "U" as suggested in the project description
X_test_xgb = process_data_xgb(X_test_raw)

X_train_cat = process_data_cat(X_train_raw) #interestingly ther doesnt seem to be  "U" as suggested in the project description
X_test_cat = process_data_cat(X_test_raw)

#%%
''' find the best hyperparameters for xgb'''
seed = 7
test_size = 0.33
X_train_xgb_h, X_test_xgb_h, y_train_xgb_h, y_test_xgb_h = model_selection.train_test_split(X_train_xgb, y_train, test_size=test_size, random_state=seed)

import xgboost as xgb 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


model_h = xgb.XGBClassifier(max_depth = 14, learning_rate=0.85, n_estimators=500,use_label_encoder=False,tree_method='exact')
model_h.fit(X_train_xgb_h, y_train_xgb_h)

# make predictions for test data

y_pred_xgb_h = model_h.predict(X_test_xgb_h)

accuracy_xgb = accuracy_score(y_test_xgb_h, y_pred_xgb_h)
f1_xgb = f1_score(y_test_xgb_h, y_pred_xgb_h)
print("Accuracy: %.5f%%" % (accuracy_xgb * 100.0))
print("Score: %.5f%%" % (f1_xgb * 100.0))


#%%
'''Run XGB on whole set for handin'''
import xgboost as xgb 

model = xgb.XGBClassifier(max_depth = 14, learning_rate=0.85, n_estimators=500,use_label_encoder=False,tree_method='exact')

model.fit(X_train_xgb,y_train)

y_pred_xgb = model.predict(X_test_xgb)


#%%
np.savetxt('xgb_jason.csv', y_pred_xgb, fmt='%i')

#%%
'''CatBoost perepare data'''
from catboost import Pool, CatBoostClassifier,cv
seed = 7
test_size = 0.33
X_train_cat_h, X_test_cat_h, y_train_cat_h, y_test_cat_h = model_selection.train_test_split(X_train_cat, y_train, test_size=test_size, random_state=seed)


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

cat_features = [0,1,2,3]


train_dataset_cat = Pool(data=X_train_cat_h, label=y_train_cat_h, cat_features = cat_features)
eval_dataset_cat = Pool(data=X_test_cat_h, label=y_test_cat_h, cat_features = cat_features)
#random_seed = 42, iterations=1000, learning_rate = 0.3, depth = 8
#%%
''' Experimenting with CV'''
'''
cv_dataset = Pool(data= X_train, label=y_train, cat_features=cat_features)

params = {"iterations": 1000,
          
          "loss_function": 'Logloss'
          }

scores = cv(cv_dataset,params, as_pandas= True )
'''

#%%
''' find best Hyperparameters for cat'''
model_h = CatBoostClassifier( cat_features=cat_features, iterations=2000, learning_rate=0.15, depth=10 ,one_hot_max_size = 20)
model_h.fit(train_dataset_cat)

# make predictions for test data

y_pred_cat_h = model_h.predict(eval_dataset_cat)

accuracy_cat = accuracy_score(y_test_cat_h, y_pred_cat_h)
f1_cat = f1_score(y_test_cat_h, y_pred_cat_h)
print("Accuracy: %.5f%%" % (accuracy_cat * 100.0))
print("Score: %.5f%%" % (f1_cat * 100.0))

#%%
from catboost import Pool, CatBoostClassifier
cat_features = [0,1,2,3]
train_dataset_cat = Pool(data=X_train_cat, label=y_train, cat_features= cat_features)
test_dataset_cat = Pool(data=X_test_cat, cat_features =cat_features)
model= CatBoostClassifier( cat_features=cat_features, iterations=2000, learning_rate=0.15, depth=10 ,one_hot_max_size = 20)
model.fit(train_dataset_cat)

y_pred_cat = model.predict(test_dataset_cat)
np.savetxt('cat_new_jason.csv', y_pred_cat, fmt='%i')


