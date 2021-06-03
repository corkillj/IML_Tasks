'''
Introduction to Machine Learning
Project - Task2

Author: Adrian Hartmann
Date: 16.04.2021

TODO: Time dependancy processing for Regression
 



'''

#%% Imports
import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
import pandas as pd 
import matplotlib.pyplot as plt


#%% Import data
train_X = pd.read_csv('train_features.csv')
train_y = pd.read_csv('train_labels.csv')

keys_12 = ['pid', 'Time', 'Age', 'EtCO2', 'PTT', 'BUN', 'Lactate', 'Temp', 'Hgb',
       'HCO3', 'BaseExcess', 'Fibrinogen', 'Phosphate', 'WBC',
       'Creatinine', 'PaCO2', 'AST', 'FiO2', 'Platelets', 'SaO2', 'Glucose',
       'Magnesium', 'Potassium', 'ABPd', 'Calcium', 'Alkalinephos', 'Bilirubin_direct', 'Chloride', 'Hct',
       'Bilirubin_total', 'TroponinI', 'ABPs', 'pH']
keys_3 = ['pid','Age','RRate', 'ABPm', 'SpO2', 'Heartrate']

#%% Preprocessing -- Extract amount of Tests made 

def process_data(X,mode):
    #get patient count
    patients = 1
    p1 = X[0,0]
    for i in range(1,X.shape[0]):
        p2 = X[i,0]
        if( p1 != p2):
            patients += 1
        p1 = p2
    #process data per patient
    if(mode == 'count'):
        X_processed = np.zeros((patients,X.shape[1]-1))
    else:
        X_processed = np.zeros((patients,X.shape[1]*1-1))
    i=0
    j=0 #pid index
    start = 0
    while(i<X.shape[0]):
        pid = X[i,0]
        while(i<X.shape[0] and X[i,0]== pid):
            i += 1
        stop = i

        if(mode == 'count'):
            addition = np.count_nonzero(~np.isnan(X[start:stop,3:]),axis=0)
        else:
            min = np.nanmean(X[start:stop,3:],axis=0)
            max = np.nanmax(X[start:stop,3:],axis=0)
            addition = max-min#np.concatenate((min,max))
        
        X_processed[j,:] = np.concatenate((X[start,(0,2)],addition))
        start = stop
        j+=1
        i+=1
    #take care of NaN
    for i in range(2,X_processed.shape[1]):
        X_processed[:,i] = np.nan_to_num(X_processed[:,i],nan=np.nanmean(X_processed[:,i]))

    return X_processed

#process data for Task3
def get_regression_data(X):
    #get patient count
    patients = 1
    p1 = X[0,0]
    for i in range(1,X.shape[0]):
        p2 = X[i,0]
        if( p1 != p2):
            patients += 1
        p1 = p2
    #process data per patient
    X3_processed = np.zeros((patients,2+8))
    i=0
    j=0 #pid index
    start = 0
    while(i<X.shape[0]):
        pid = X[i,0]
        while(i<X.shape[0] and X[i,0]== pid):
            i += 1
        stop = i
        data1 = np.nanmean(X[start:start+6,2:],axis=0)
        data2 = np.nanmean(X[stop-6:stop,2:],axis=0)
        data = np.concatenate((data1,data2),axis=0)
        #data = X[start:stop,2:]
        X3_processed[j,:] = np.concatenate((X[start,(0,1)],data.flatten(order='F')))
        
        start = stop
        j+=1
        i+=1
    #take care of NaN
    for i in range(2,X3_processed.shape[1]):
        X3_processed[:,i] = np.nan_to_num(X3_processed[:,i],nan=np.nanmean(X3_processed[:,i]))


    return X3_processed

X_processed = process_data(train_X.to_numpy(), 'count')
scaler = preprocessing.StandardScaler().fit(X_processed[:,1:])
X_processed[:,1:] = scaler.transform(X_processed[:,1:])

#X2_processed = process_data(train_X.to_numpy(), 'minmax')


X3_processed = get_regression_data(train_X[keys_3].to_numpy())

#%% Task 1
'''
X values
'pid', 'Time', 'Age','BaseExcess', 'Fibrinogen', 'AST', 'Alkalinephos',
'Bilirubin_total','Lactate', 'TroponinI', 'SaO2','Bilirubin_direct', 'EtCO2'
'''
y_1 = train_y[['pid','LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 
            'LABEL_Bilirubin_total','LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
            'LABEL_Bilirubin_direct', 'LABEL_EtCO2']].to_numpy()


def train_model(X,y):
    model = svm.SVC(probability=True,cache_size=2000,C=0.2,class_weight='balanced')
    model.fit(X,y)
    return model

#%%
print("Starting SVMs for Task1")
models = []

for i in range(1,11):
    models.append(train_model(X_processed[:,1:],y_1[:,i]))
    y1_pred = models[-1].predict_proba(X_processed[:,1:])[:,1]
    print("ROC: ", metrics.roc_auc_score(y_1[:,i], y1_pred))

    print("Finished Model ",i)


print("done")

#%% Task 2
print("Starting SVM for Task 2")
y_2 = train_y[['pid','LABEL_Sepsis']].to_numpy()

model_task2 = train_model(X_processed[:,1:],y_2[:,1])
y2_pred = model_task2.predict_proba(X_processed[:,1:])[:,1]

print("Accuracy ROC: ", metrics.roc_auc_score(y_2[:,1], y2_pred))



#%% Task 3 - Regression
y_3 = train_y[['pid','LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']]
y3 = y_3.to_numpy()

print("Starting Task 3 Regressions")

#X3_processed = get_regression_data(train_X[keys_3].to_numpy())
#X3_processed = process_data(train_X.to_numpy(),'minmax')

al = 0.1
rrate = linear_model.Ridge(alpha=al)
rrate.fit(X3_processed[:,1:],y3[:,1])
y3_pred = rrate.predict(X3_processed[:,1:])
print("R2-Score: ", np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(y3[:,1], y3_pred))]))

abpm = linear_model.Ridge(alpha=al)
abpm.fit(X3_processed[:,1:],y3[:,2])
y3_pred = abpm.predict(X3_processed[:,1:])
print("R2-Score: ", np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(y3[:,2], y3_pred))]))

sp02 = linear_model.Ridge(alpha=al)
sp02.fit(X3_processed[:,1:],y3[:,3])
y3_pred = sp02.predict(X3_processed[:,1:])
print("R2-Score: ", np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(y3[:,3], y3_pred))]))

heartrate = linear_model.Ridge(alpha=al)
heartrate.fit(X3_processed[:,1:],y3[:,4])
y3_pred = heartrate.predict(X3_processed[:,1:])
print("R2-Score: ", np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(y3[:,4], y3_pred))]))

print("done")

# %% build output

test_set = pd.read_csv('test_features.csv').to_numpy()
test_set3 = pd.read_csv('test_features.csv').get(keys_3).to_numpy()

#%% Task 1 on test set
X1_test = process_data(test_set,'count')
X1_test[:,1:] = scaler.transform(X1_test[:,1:])

output = np.zeros((X1_test.shape[0],16))
output[:,0] = X1_test[:,0]

for i in range(len(models)):
    output[:,i+1] = models[i].predict_proba(X1_test[:,1:])[:,1]
print("done")

#%%Task 2 on test set
X2_test = X1_test

output[:,11] = model_task2.predict_proba(X2_test[:,1:])[:,1]

#%% Task 3 on test set
print("Processing...")
X3_test = get_regression_data(test_set3)

output[:,12] = rrate.predict(X3_test[:,1:])
output[:,13] = abpm.predict(X3_test[:,1:])
output[:,14] = sp02.predict(X3_test[:,1:])
output[:,15] = heartrate.predict(X3_test[:,1:])
print("done")

#%% Save output
head = "pid,LABEL_BaseExcess,LABEL_Fibrinogen,LABEL_AST,LABEL_Alkalinephos,LABEL_Bilirubin_total,LABEL_Lactate,LABEL_TroponinI,LABEL_SaO2,LABEL_Bilirubin_direct,LABEL_EtCO2,LABEL_Sepsis,LABEL_RRate,LABEL_ABPm,LABEL_SpO2,LABEL_Heartrate"
np.savetxt('result_Task2.csv',output,delimiter= ',',fmt='%.3f',header=head,comments="")
print("saved")
# %%
