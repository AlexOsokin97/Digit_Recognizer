# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:41:40 2020

@author: User
"""

import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')
data.drop(['label'], axis=1, inplace=True)

data['avg_pixel_used'] = np.nan

for i in range(len(data)):
    
    val = data.iloc[[i], :-1]
    counter = 0
        
    for col in val.columns:
    
        if val[col].unique() > 0:
                
            counter += 1
                
        else: 
            continue
        
    data.loc[[i], 'avg_pixel_used'] = counter/784
   
###############################################################################################

data['max_pixel_val'] = np.nan

img = data.iloc[:, :-2].values.reshape(-1, 28, 28, 1)/255.0

test = img[0][:14,:14,0]
test2 = img[0][14:,:14,0]

for i in range(len(img)):
    
    sub_img1 = np.average(img[i][:14,:14,0])
    sub_img2 = np.average(img[i][14:,:14,0])
    sub_img3 = np.average(img[i][:14,14:,0])
    sub_img4 = np.average(img[i][14:,14:,0])
    
    maximum = np.max([sub_img1,sub_img2,sub_img3,sub_img4])
    
    data.loc[[i], 'max_pixel_val'] = maximum
    
################################################################################################
    
data['avg_pixel_val'] = np.nan

for i in range(len(data)):
    
    pixels = []
    
    for col in data.iloc[[i], :-3].columns:
        
        pixels.append(col.values)
    
    data.loc[[i], 'avg_pixel_val'] = np.average(pixels)
                         
                         
                         
                         
                         
                         