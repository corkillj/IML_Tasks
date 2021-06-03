#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:51:13 2021

@author: mariankannwischer
"""

#%%
import pandas as pd
import numpy as np

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


from sklearn.preprocessing import OneHotEncoder
def preprocess_data_onehot(X):
    encoder = OneHotEncoder()
    X_proc = preprocess_data(X)
    X_oh = encoder.fit_transform(X_proc).toarray()
    categories = [chr(num) for num in encoder.categories_[0]]
    columns = []
    for i in range(4):
        columns.extend([char + str(i) for char in categories])
    return pd.DataFrame(X_oh, columns=columns)

    
#%%
for i, sample in X.iterrows():

#%%
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_oh = encoder.fit_transform(X_proc).toarray()

#%%
X_oh = preprocess_data_onehot(X_train)

#%%
X_proc = preprocess_data(X_train)

#%%
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# for par in [80, 120]:
clf = xgb.XGBClassifier(max_depth=10, eta=0.5, gamma=0.4, n_estimators=100, verbosity=0, use_label_encoder=False)
print('---' + str(np.mean(cross_validate(estimator=clf, X=X_proc, y=y_train.values, cv=10, scoring='f1')['test_score'])) + '---')


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
y_sub.to_csv('pred_mari_xgb2.csv', index=False, header=False)

#%% Create DataSet
class ProteinMutationDataset(torch.utils.data.Dataset):
    def __init__(self, X)

#%% Setup Network
import torch
from torch import nn
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(80, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )

    def forward(self, x):
        pred = self.linear_relu_stack(x)
        return pred

model = NeuralNetwork().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#%% Train Network
    

for epoch in range(1):
    
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        runnin_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
print('Finished Training')
    