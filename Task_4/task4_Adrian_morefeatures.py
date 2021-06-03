'''
    Introduction to Machine Learning - Task 4

    Author: Adrian Hartmann

'''
#%% nothing
""" import matplotlib.pyplot as plt  #use once to determine smalles image size
y = 10000
x = 10000
for name in image_names:
    img = plt.imread("food/"+name)
    if(y > img.shape[0]):
        y = img.shape[0]
    
    if(x > img.shape[1]):
        x = img.shape[1]
print("x_min: ",x,"| y_min: ",y) # x_min = 354 | y_min:  242 """
#%%
import torch
import os
from torch.utils.data import Dataset, DataLoader
#from torchvision.io import read_image
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

path = Path('food-101')
path_meta = path /'meta'
path_images = path/'images'
classes_path = path_meta/'classes.txt'
labels_txt = pd.read_csv(classes_path, header=None)

#%%
def build_data_frame(path_name, file_name, img_format='jpg'):
    """
    build_data_frame input the path and file name, the function will return the dataframe with two columns:
    ['label'] : image label
    ['image_file'] : image file name with directory information
    input paramters:
    path_name : path
    file_name : file name, string
    img_format : default format is jpg

    return dataframe
    """
    path_file = path_name / file_name
    file_df = pd.read_csv(path_file, delimiter='/', header=None, names=['label', 'image_file'])
    file_df['image_file'] = file_df['label'].astype(str) + '/' + file_df['image_file'].astype(str) + '.' + img_format

    return file_df


#%%

train_df = build_data_frame(path_meta, 'train.txt')
test_df = build_data_frame(path_meta, 'test.txt')

train_df = train_df.sample(frac=1).reset_index(drop=True)[0:5000]
test_df = pd.DataFrame(test_df.sample(frac=1).reset_index(drop=True)[0:50])
#%%
class Dataset2(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 256
            transforms.CenterCrop(224),  # 224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open('food-101/images/' + self.df.loc[index,'image_file']).convert('RGB')
        X = self.transform(X)
        y = labels_txt[labels_txt[0] == self.df.loc[index,'label']].index.values[0]

        return X, y
#%%
class Dataset(torch.utils.data.Dataset):
  def __init__(self, id_list):
      self.id_list = id_list
      self.transform = transforms.Compose([
            transforms.Resize(256),  #256
            transforms.CenterCrop(224), #224
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
image101_train_set = Dataset2(train_df)
image101_test_set = Dataset2(test_df)
image_set = Dataset(image_names)
image_loader = DataLoader(image_set, batch_size=1)
image101_train_loader = DataLoader(image101_train_set, batch_size=8)
image101_test_loader = DataLoader(image101_test_set,batch_size=4)
# %% show first image 

#feature_batch , label = next(iter(image_loader)) 
#plt.imshow(feature_batch[9].permute((1,2,0))) ### remove normalization for true image
# %% Classify all Images

import torch
from torch import nn
import torchvision.models as models
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
#model.fc = torch.nn.Linear(model.fc.in_features, 101)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 101),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.03)
model.to(device)

#%%
epochs = 1
steps = 0
running_loss = 0
print_every = 50
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in image101_train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(steps)
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in image101_test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(image101_train_loader))
            test_losses.append(test_loss / len(image101_test_loader))
            print(f"Epoch {epoch + 1}/{epochs}.. "
                    f"Train loss: {running_loss / print_every:.3f}.. "
                    f"Test loss: {test_loss / len(image101_test_loader):.3f}.. "
                    f"Test accuracy: {accuracy / len(image101_test_loader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'model.pth')


#%%
print("Classify Images with ResNet")
print("Progress:")

classification = np.zeros((len(image_names),1000))
for  batch, idx in image_loader:
    if idx.numpy()[0]%500 == 0:
        print("-",end='')
    if torch.cuda.is_available():
        batch = batch.to('cuda')

    with torch.no_grad():
        image_class = model(batch)

    probabilities = torch.nn.functional.softmax(image_class[0], dim=0)
    
    #EXTRACT the probabilities for trainingsset
    #top_prob, top_catid = torch.topk(probabilities, 4)
    classification[idx,:] = probabilities.cpu().numpy() #torch.argmax(probabilities).cpu().numpy()
    #classification[idx,4:8] = top_catid.cpu().numpy() #torch.max(probabilities).cpu().numpy()

print("done")

#%% BUILD trainingset with classification output
import random
random.seed(42)
train_txt = np.genfromtxt('train_triplets.txt',delimiter=' ').astype(int)

train_X = np.zeros((train_txt.shape[0],3000))
train_y = np.zeros((train_txt.shape[0],1))

for i in range(0,train_X.shape[0]):
    train_X[i,0:1000] = classification[train_txt[i,0],:]
    if(random.random()>0.5):
        train_X[i,1000:2000] = classification[train_txt[i,1],:]
        train_X[i,2000:3000] = classification[train_txt[i,2],:]
        train_y[i] = 1
    else:
        train_X[i,1000:2000] = classification[train_txt[i,2],:]
        train_X[i,2000:3000] = classification[train_txt[i,1],:]
        train_y[i] = 0

#%% train XGBoost on given labels
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(train_X, train_y, random_state=12,test_size = 0.50)

split = 1700
X_te = train_X[np.max(train_txt,axis=1)<split]
y_te = train_y[np.max(train_txt,axis=1)<split]
X_tr = train_X[np.min(train_txt,axis=1)>=split]
y_tr = train_y[np.min(train_txt,axis=1)>=split]

model = xgb.XGBClassifier(tree_method = 'auto',n_estimators = 500, \
                         max_depth = 8,use_label_encoder=False, eta=0.2)
model.fit(X_tr,y_tr)


y_pred = model.predict(X_te)
print("Accuracy:",accuracy_score(y_te,y_pred))
print("Accuracy train set:",accuracy_score(y_tr,model.predict(X_tr)))


#%% train on full data and predict output
#model.fit(train_X,train_y)
#print("Accuracy total:",accuracy_score(train_y,model.predict(train_X)))

test_txt = np.genfromtxt('test_triplets.txt',delimiter=' ').astype(int)

test_X = np.zeros((test_txt.shape[0],6))

for i in range(0,test_X.shape[0]):
    test_X[i,0:1000] = classification[test_txt[i,0],:]
    test_X[i,1000:2000] = classification[test_txt[i,1],:]
    test_X[i,2000:3000] = classification[test_txt[i,2],:]



y_output = model.predict(test_X)
# %%
np.savetxt('task4_adi.txt',y_output,fmt="%i")

# %%
