# import pandas as pd
# import numpy as np
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D  # convolution layers
from keras.layers import Dense, Dropout, Flatten  # core layers

from keras.preprocessing.image import ImageDataGenerator

# from keras.datasets import mnist

import os

print(os.listdir('dataset'))  # ['sample_submission.csv', 'test.csv', 'train.csv']

# LOAD DATA
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print("Data are Ready!!")

# PRINT DATA SIZE
print(f"Training data size is {train.shape}\nTesting data size is {test.shape}")

# SET DATA FEATURES AND LABELS
X = train.drop(['label'], 1).values
y = train['label'].values