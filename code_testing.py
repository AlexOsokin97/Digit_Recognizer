# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:41:40 2020

@author: User
"""

import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')

zeros = data.loc[data['label'] == 0]
zeros.drop(['label'], axis=1, inplace=True)

cols = zeros.columns

zero1 = zeros.iloc[[1], 1:]

image = []
i = 0
col_list = zero1.columns

while i < 28:
    row = []
    counter = 0
    for column in col_list:
        val = zero1[column].values[0]
        row.append(val)
        counter += 1
        col_list.remove(column)
        if counter == 27:
            break
    image.append(row)
    i += 1

