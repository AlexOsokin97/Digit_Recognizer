# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:41:40 2020

@author: User
"""

import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')

#############################################################################

def avg_pixel_used(data, digits=[]):
    
    pixel_used = {}
    
    for dig in digits:
        
        val = data.loc[data['label']==digits[dig]].iloc[[0], 1:]
        counter = 0
        
        for col in val.columns:
            
            if val[col].unique() > 0:
                
                counter += 1
                
            else: 
                continue
        
        pixel_used[dig] = counter/784

    return pixel_used

##############################################################################

test = avg_pixel_used(data, [x for x in range(10)])

data_copy = data.copy()

data_copy['avg_pixel_used'] = np.nan

##############################################################################

def fill_avgPixelUsed(data, digits=[]):
    
    for dig in digits:
        
        data.loc[data['label'] == dig, 'avg_pixel_used'] = test[dig]

##############################################################################

fill_avgPixelUsed(data_copy, [x for x in range(10)])

data_copy['avg_pixel_used'].value_counts()

data_copy['avg_pixel_val'] = np.nan

#############################################################################

def fill_avgPixelVal(data, digits=[]):
    
    for dig in digits:
        
        data.loc[data['label'] == dig, 'avg_pixel_val'] = test[dig]

##############################################################################