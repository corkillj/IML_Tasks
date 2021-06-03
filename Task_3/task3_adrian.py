'''
Introduction to Machine Learning
Project - Task3

Author: Adrian Hartmann
Date: 04.05.2021

'''

#%% Imports
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train['Active']
# %%
X = train['Sequence'].apply(lambda x: pd.Series(list(x)))#.applymap(ord)

#%%
enc = OneHotEncoder(sparse=False).fit(X)
X_enc = enc.transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_enc, y, random_state=1724,test_size = 0.25)
# %%
import xgboost as xgb

model = xgb.XGBClassifier(n_jobs=-1,n_estimators =300,use_label_encoder=False,max_depth=12)
model.fit(X_tr,y_tr)

print('done fitting')
y_tr_pred = model.predict(X_tr)
y_te_pred = model.predict(X_te)

print("Train Score:", f1_score(y_tr, y_tr_pred))
print("Validation Score:", f1_score(y_te, y_te_pred))
print('done')
# %% output

model.fit(X_enc,y)

X_test = enc.transform(test['Sequence'].apply(lambda x: pd.Series(list(x))))

output = model.predict(X_test)

# %% save

np.savetxt('result_Task3_adi.csv',output,delimiter= ',',fmt='%i',header="",comments="")

# %%
