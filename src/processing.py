import numpy as np
import cv2 as cv

def preprocess(img:np.array):
    '''
    Used for both training and jetbot preprocessing.
    '''
    return img/255