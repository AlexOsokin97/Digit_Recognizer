# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:21:17 2020

@author: User
"""
import pandas as pd
import numpy as np
import pickle
import transformation

data = pd.read_csv('../test.csv')
features = transformation.data_transformation(data)

loaded_model = pickle.load(open('../Models/SVM_98%_Kaggle.sav', 'rb'))
pred = loaded_model.predict(features)





