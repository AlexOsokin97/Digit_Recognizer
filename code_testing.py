# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:41:40 2020

@author: User
"""

import pandas as pd
import numpy as np
from PIL import Image

data = pd.read_csv('train.csv')

zeros = data.loc[data['label'] == 0].copy()
zeros.drop(['label'], axis=1, inplace=True)
zeros.reset_index(drop=True, inplace=True)

ones = data.loc[data['label'] == 1].copy()
ones.drop(['label'], axis=1, inplace=True)
ones.reset_index(drop=True, inplace=True)

cols = ones.columns.tolist()

one1 = ones.iloc[[1], :].copy()

one1_pixls = []
i = 0
col_list = one1.columns.tolist()

while i < 28:
    row = []
    counter = 0
    for column in col_list:
        val = one1[column].values[0]
        row.append(val)
        counter += 1
        col_list.remove(column)
        if counter == 27:
            break
    one1_pixls.append(row)
    i += 1

one1_pixls = np.array(one1_pixls, dtype=np.uint8)
image = Image.fromarray(one1_pixls)
image.save('new.png')









