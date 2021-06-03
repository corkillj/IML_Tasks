'''
    Introduction to Machine Learning - Task 4

    Author: Adrian Hartmann

    TODO: 
'''

#%%
from numpy.ma.core import concatenate
import torch
from torch import nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision.io import read_image
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler  

class Dataset(torch.utils.data.Dataset):
  def __init__(self, id_list):
      self.id_list = id_list
      self.transform = transforms.Compose([
            transforms.Resize((256,256)),  #256
            #transforms.CenterCrop(256), #224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
      

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.id_list)

  def __getitem__(self, index):
        X = Image.open('food/' + self.id_list[index])
        X = self.transform(X)
        y = index

        return X, y

# %%
image_names = os.listdir("food")
image_set = Dataset(image_names)
image_loader = DataLoader(image_set, batch_size=1)
# %% show first image 

#feature_batch , label = next(iter(image_loader)) 
#plt.imshow(feature_batch[9].permute((1,2,0))) ### remove normalization for true image
# %% Classify all Images
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

import torch
#model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)

# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
num_ftrs = model.fc.in_features
model.fc = Identity()
model.eval()

if torch.cuda.is_available():
    model.to('cuda')

#%%
print("Classify Images with ResNet")
print("Progress:")

features = np.zeros((len(image_names),num_ftrs))
for  batch, idx in image_loader:
    if idx.numpy()[0]%500 == 0:
        print("-",end='')
    if torch.cuda.is_available():
        batch = batch.to('cuda')

    with torch.no_grad():
        image_class = model(batch)

    #just save all output features
    features[idx,:] = image_class.cpu().numpy()

print("done")

#%% BUILD trainingset with classification output
import random
random.seed(42)
train_txt = np.genfromtxt('train_triplets.txt',delimiter=' ').astype(int)

train_X = np.zeros((train_txt.shape[0],3*num_ftrs))
train_y = np.zeros((train_txt.shape[0],1))

for i in range(0,train_X.shape[0]):
    train_X[i,0:num_ftrs] = features[train_txt[i,0],:]
    if(random.random()>0.5):
       train_X[i,num_ftrs:2*num_ftrs] = features[train_txt[i,1],:]
       train_X[i,2*num_ftrs:3*num_ftrs] = features[train_txt[i,2],:]
       train_y[i] = 1
    else:
       train_X[i,num_ftrs:2*num_ftrs] = features[train_txt[i,2],:]
       train_X[i,2*num_ftrs:3*num_ftrs] = features[train_txt[i,1],:]
       train_y[i] = 0

#split up in train & test with different source images
split = 1600
X_te = train_X[np.max(train_txt,axis=1)<split]
y_te = train_y[np.max(train_txt,axis=1)<split]
X_tr = train_X[np.min(train_txt,axis=1)>=split]
y_tr = train_y[np.min(train_txt,axis=1)>=split]

#augment data with swapped 2nd & 3rd image
X_te = np.concatenate((X_te,np.concatenate((X_te[:,0:num_ftrs],X_te[:,2*num_ftrs:3*num_ftrs],X_te[:,num_ftrs:2*num_ftrs]),axis=1)),axis = 0)
X_tr = np.concatenate((X_tr,np.concatenate((X_tr[:,0:num_ftrs],X_tr[:,2*num_ftrs:3*num_ftrs],X_tr[:,num_ftrs:2*num_ftrs]),axis=1)),axis = 0)
y_te = np.concatenate((y_te,1-y_te))
y_tr = np.concatenate((y_tr,1-y_tr))



#%% build NN stuff
# Source: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89

#Neural Network Model
class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is num_ftrs*3
        self.layer_1 = nn.Linear(3*num_ftrs, 177) 
        self.layer_2 = nn.Linear(177, 76)
    
        self.layer_out = nn.Linear(76, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)

        self.batchnorm1 = nn.BatchNorm1d(177)

        
    def forward(self, inputs):
        x = self.dropout(inputs)
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.layer_out(x)
        
        return x

#Accuracy metric
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = acc * 100
    
    return acc

#%% Train
EPOCHS = 42
BATCH_SIZE = 1024
torch.manual_seed(69)

train_dataset = TensorDataset(torch.FloatTensor(X_tr),torch.FloatTensor(y_tr.squeeze()))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = binaryClassification()
model.to(device)
#print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.01)

#begin model training
best_score = 0
for e in range(1, EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    #Calculate Validation Error
    model.eval()
    val_score = 0
    with torch.no_grad():
        X_test = torch.FloatTensor(X_te).to(device)
        y_pred_val = model(X_test)
        val_score = binary_acc(y_pred_val, torch.FloatTensor(y_te.squeeze()).unsqueeze(1).to(device)).item()
    
    #save best model weights
    if(val_score > best_score):
        best_model = copy.deepcopy(model.state_dict())
        best_score = val_score
    
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f} | Val: {val_score:.3f}')

#load best model
model.load_state_dict(best_model)
print("BestValue: ",best_score)

#%% check validation score
model.eval()
with torch.no_grad():
    y_pred_val = model(torch.FloatTensor(X_te).to(device)).cpu()
    y_pred_val = torch.round(torch.sigmoid(y_pred_val)).numpy()

print(np.sum(y_pred_val == y_te)/y_te.shape[0])


#%% predict output
#load test samples and assign the features generated from ResNet
test_txt = np.genfromtxt('test_triplets.txt',delimiter=' ').astype(int)

test_X = np.zeros((test_txt.shape[0],3*num_ftrs))
for i in range(0,test_X.shape[0]):
    test_X[i,0:num_ftrs] = features[test_txt[i,0],:]
    test_X[i,num_ftrs:2*num_ftrs] = features[test_txt[i,1],:]
    test_X[i,2*num_ftrs:3*num_ftrs] = features[test_txt[i,2],:]

model.eval()
with torch.no_grad():
    y_out = model(torch.FloatTensor(test_X).to(device)).cpu()
    y_out = torch.round(torch.sigmoid(y_out)).numpy()


# %%
np.savetxt('task4_adi_nn_7285.txt',y_out,fmt="%i")


# %%
