"""
Data Augmentation using ImageDataGenerator

Data augmentation is a technique used to artificially expand the size of a dataset by applying
random transformations such as rotations, shifts, zooms, flips, and other alterations to the input
images. This helps improve the generalization ability of machine learning models, especially for
tasks like image classification, by introducing more variety into the training set.

This particular configuration of `ImageDataGenerator` applies a set of transformations to the input
images for augmentation during training.

Parameters:
- featurewise_center (False): If set to True, this option computes the mean of the dataset and
  subtracts it from each input image, ensuring the mean of the dataset becomes 0. Here, it's set
  to False, meaning no mean centering is applied globally across the dataset.

- samplewise_center (False): If set to True, each input image will have its mean subtracted to center
  it around 0. In this case, it's set to False, so no sample-wise centering is performed.

- featurewise_std_normalization (False): If True, it normalizes each input by dividing it by the standard
  deviation of the entire dataset. Here, it's set to False, meaning no dataset-wide standardization is
  applied.

- samplewise_std_normalization (False): If True, each input image will be normalized by its own standard
  deviation. Here, it's False, meaning no per-sample standardization is done.

- zca_whitening (False): If set to True, ZCA whitening is applied to the dataset, which decorrelates the
  features by removing linear dependencies between pixels. It's False here, so ZCA whitening is not applied.

- rotation_range (15): The input images will be randomly rotated within a range of -15 to +15 degrees,
  which introduces variety in image orientation.

- zoom_range (0.05): Random zooming is applied to the input images, where the zoom factor can vary by
  up to 1%. This adds small variations in scale.

- width_shift_range (0.1): The images will be randomly shifted horizontally by up to 10% of the total
  image width. This introduces randomness in the horizontal position of the image.

- height_shift_range (0.1): The images will be randomly shifted vertically by up to 10% of the total
  image height. This introduces randomness in the vertical position of the image.

- horizontal_flip (False): If set to True, images would be randomly flipped horizontally. It's set
  to False here, so no horizontal flipping will occur.

- vertical_flip (False): If set to True, images would be randomly flipped vertically. It's set
  to False here, so no vertical flipping will occur.

This configuration applies several augmentations like random rotations, zooms, and shifts to increase
the variety of training samples without distorting the dataset too much.
"""

from keras_preprocessing.image import ImageDataGenerator
from data_manipulation import X_train, y_train, X_test, y_test

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.05,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# datagen.fit(X_train)
train_gen = datagen.flow(X_train, y_train, batch_size=128)
test_gen = datagen.flow(X_test, y_test, batch_size=128)