import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D  # convolution layers
from keras.layers import Dense, Dropout, Flatten  # core layers

from keras.preprocessing.image import ImageDataGenerator

# from keras.datasets import mnist

import os

print(os.listdir('dataset'))  # ['sample_submission.csv', 'test.csv', 'train.csv']

# LOAD DATA
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
sub = pd.read_csv('dataset/sample_submission.csv')
print("Dataset is Loaded!!")

# PRINT DATA SIZE
print(f"Training data size is {train.shape}\nTesting data size is {test.shape}")

# SET DATA FEATURES AND LABELS

# 1. Set x, y format - X = number images, Y = associated labels
X = train.drop(['label'], axis=1).values  # drop label column for x data, keep only pixel columns
y = train['label'].values

# 2. Normalize and reshape values
X = X.astype('float64')
X /= 255.0  # Normalize X values between [0,1] - grayscale is from 0 to 255
X = X.reshape(-1, 28, 28, 1)  # Reshape X array so that each example will be grayscale image of 28 x 28

# 3. One-hot encoding for Y labels (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y = to_categorical(y)  # Labels are 10 digits numbers from 0 to 9. We need to encode these labels to one hot vectors

# 4. Split training set and validation set
# Test_size = 0.1 means 10% of data is for testing, 90% is for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# shape of data: ((37800, 28, 28, 1), (4200, 28, 28, 1), (37800, 10), (4200, 10))


# DATA VISUALIZATION
X_train__ = X_train.reshape(X_train.shape[0], 28, 28)

fig, axis = plt.subplots(1, 4, figsize=(20, 10))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_train__[i], cmap='binary')
    digit = y_train[i].argmax()
    ax.set(title=f"Real Number is {digit}")
#
# # NORMALIZATION
mean = np.mean(X_train)
std = np.std(X_train)

def standardize(x):
    return (x - mean) / std
