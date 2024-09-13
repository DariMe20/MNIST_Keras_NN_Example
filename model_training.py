import keras
import numpy as np
from keras.src.utils import plot_model
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from data_augmentation import train_gen, test_gen
from data_manipulation import X_train, X_test, X, y, y_test
from keras_model import model

epochs = 30
batch_size = 128
train_steps = X_train.shape[0] // batch_size
valid_steps = X_test.shape[0] // batch_size

# BASIC MODEL FIT (TRAINING)
# model.fit(X, y, batch_size=batch_size, validation_split=0.2, epochs=10)

"""ADJUSTED MODEL FIT """
# es - Early stopping callback - if accuracy value doesn't improve over 10 consecutive epochs, training stops
es = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",  # metrics to monitor
    patience=10,  # how many epochs before stop
    verbose=1,  # make it talk
    mode="max",  # we need the maximum accuracy.
    restore_best_weights=True,  # after training, model automatically restores best metrics
    )

# rp = ReduceLROnPlateau Callback
rp = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",  # analyse validation accuracy
    factor=0.2,  # reduce lr with 20% when validation accuracy performance decreases
    patience=3,  # over 3 consecutive epochs
    verbose=1,
    mode="max",  # we need the best value
    min_lr=0.00001,  # minimum lr - it cannot decrease anymore than this value
    )

# Fit the model - save data in history variable
history = model.fit(train_gen,
                    epochs=epochs,
                    steps_per_epoch=train_steps,
                    validation_data=test_gen,
                    validation_steps=valid_steps,
                    callbacks=[es, rp])

"""............PLOTTING FOR RESULTS AND METRICS"""
plot_model(model, to_file='CNN_model_arch.png', show_shapes=True, show_layer_names=True)

"""TRAINING AND VALIDATION CURVES"""
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1, figsize=(18, 10))

# Plot training & validation loss
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='best', shadow=True)

# Plot training & validation accuracy
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()

"""CONFUSION MATRIX"""
fig = plt.figure(figsize=(10, 10))  # Set Figure

y_pred = model.predict(X_test)  # Predict class probabilities as 2 => [0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]

Y_pred = np.argmax(y_pred, 1)  # Decode Predicted labels
Y_test = np.argmax(y_test, 1)  # Decode labels

mat = confusion_matrix(Y_test, Y_pred)  # Confusion matrix

# Plot Confusion matrix
heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='.0f')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()

"""PREDICTIONS ON TEST DATA"""
y_pred = model.predict(X_test)
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title=f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}")
