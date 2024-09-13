import os
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
from seaborn import heatmap
from data_manipulation import X_test, y_test

# Set the model name (can be dynamic based on your requirements)
model_name = 'trained_model'
folder_path = f'./{model_name}_plots/'

# Create the main directory for the model plots
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Load the trained model
model = load_model(f'{model_name}.h5')

# Load training history
with open(f'history/history.pkl', 'rb') as f:
    history = pickle.load(f)

# Create subfolders for different types of plots
os.makedirs(f'{folder_path}/training_curves/', exist_ok=True)
os.makedirs(f'{folder_path}/confusion_matrix/', exist_ok=True)
os.makedirs(f'{folder_path}/predictions/', exist_ok=True)

"""TRAINING AND VALIDATION CURVES"""
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1, figsize=(18, 10))

# Plot training & validation loss
ax[0].plot(history['loss'], color='b', label="Training loss")
ax[0].plot(history['val_loss'], color='r', label="Validation loss")
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='best', shadow=True)

# Plot training & validation accuracy
ax[1].plot(history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='best', shadow=True)

plt.tight_layout()
# Save the plot
plt.savefig(f'{folder_path}/training_curves/training_validation_curves.png')


"""CONFUSION MATRIX"""
y_pred = model.predict(X_test)
Y_pred = np.argmax(y_pred, axis=1)
Y_test = np.argmax(y_test, axis=1)

mat = confusion_matrix(Y_test, Y_pred)

# Plot Confusion Matrix
fig = plt.figure(figsize=(10, 10))
heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='.0f')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')

# Save the confusion matrix plot
plt.savefig(f'{folder_path}/confusion_matrix/confusion_matrix.png')
plt.show()

"""PREDICTIONS ON TEST DATA"""
y_pred = model.predict(X_test)
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

# Plot predicted vs actual for a few examples
fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}")

# Save the predictions plot
plt.savefig(f'{folder_path}/predictions/predictions_on_test_data.png')
plt.show()

print("Plots saved successfully!")
