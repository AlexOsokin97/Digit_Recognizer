# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:12:12 2020

@author: Alexander
"""

from PIL import Image
import cv2
import base64
import numpy as np
from io import BytesIO


def getImage(data):
    
    data_url = data
    width, height = 28, 28
    
    image_64 = data_url[22:]
    binary = base64.b64decode(image_64)
    image = np.asarray(bytearray(binary), dtype="uint8")
    #img = Image.fromarray(data, 'RGB')
    
    return image
    
    

