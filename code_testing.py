# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:41:40 2020

@author: User
"""

import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')


zeros = data.loc[data['label'] == 0].copy()
zeros.drop(['label'],axis=1, inplace=True)

zeros = zeros/255.0

zeros_img_mode = zeros.values.reshape(-1, 28, 28, 1)

np.average(zeros_img_mode[0][:,:,0])

ones = data.loc[data['label'] == 1].copy()
ones.drop(['label'],axis=1, inplace=True)

ones = ones/255.0

ones_img_mode = ones.values.reshape(-1, 28, 28, 1)

np.average(ones_img_mode[0][:,:,0])

img = data.loc[data['label'] == 1].iloc[:, 1:].values.reshape(-1, 28, 28, 1)

def average_pixel_value(data, digits=[]):
    
    avg_pixel = {}
    
    for dig in digits:
        
        img = data.loc[data['label'] == digits[dig]].iloc[:, 1:].values.reshape(-1, 28, 28, 1)
        #img = dig_loc.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
        avg = np.average(img[0][:,:,0])
        avg_pixel[dig] = avg
    
    return avg_pixel

arr = [x for x in range(0,10,1)]

dic = average_pixel_value(data, [x for x in range(0,10,1)])

dic = pd.Series(dic).to_frame('Average_pixel')

dic = dic/255.0







