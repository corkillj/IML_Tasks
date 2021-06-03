import torch
import os
from torch.utils.data import Dataset, DataLoader
# from torchvision.io import read_image
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

path = Path('food-101')
path_meta = path / 'meta'
path_images = path / 'images'
classes_path = path_meta / 'classes.txt'
labels_txt = pd.read_csv(classes_path, header=None)


# %%
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


# %%

train_df = build_data_frame(path_meta, 'train.txt')
test_df = build_data_frame(path_meta, 'test.txt')

train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = pd.DataFrame(test_df.sample(frac=1).reset_index(drop=True)[0:101])


# %%
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
        X = Image.open('food-101/images/' + self.df.loc[index, 'image_file']).convert('RGB')
        X = self.transform(X)
        y = labels_txt[labels_txt[0] == self.df.loc[index, 'label']].index.values[0]

        return X, y


# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, id_list):
        self.id_list = id_list
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 256
            transforms.CenterCrop(224),  # 224
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
image101_train_loader = DataLoader(image101_train_set, batch_size=16)
image101_test_loader = DataLoader(image101_test_set, batch_size=1)
# %% show first image

# feature_batch , label = next(iter(image_loader))
# plt.imshow(feature_batch[9].permute((1,2,0))) ### remove normalization for true image
# %% Classify all Images

import torch
from torch import nn
import torchvision.models as models

model = models.vgg16(pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")
# model.fc = torch.nn.Linear(model.fc.in_features, 101)


model.fc = nn.Linear(2048, 101)


criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

# %%
epochs = 1
steps = 0
running_loss = 0
print_every = 1000
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
        # print(steps)
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
