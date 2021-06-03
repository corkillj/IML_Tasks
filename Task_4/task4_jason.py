# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:39:50 2021

@author: Jason
"""
#%%

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


path = vision.Path('food-101')
path_meta = path /'meta'
path_images = path/'images'



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

#%%

bs = 12
train_model_data = (ImageList.from_df(df=train_df,path=path_images, cols=1)\
                            .split_by_rand_pct(0.2)\
                            .label_from_df(cols=0)\
                            .transform( size=224)\
                            .databunch(bs=bs)\
                            .normalize(imagenet_stats))
#%%

torch.cuda.get_device_name(0)
#%%
top_5_accuracy = partial(top_k_accuracy, k=5)

learn = cnn_learner(train_model_data, models.resnet50, metrics=[accuracy, top_5_accuracy], callback_fns=ShowGraph)
#%%

learn.fit_one_cycle(4, max_lr=slice(1e-9, 1e-4))
learn.save('food-101-ResNet50')

#%%
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
    
#%%
