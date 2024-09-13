import os
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
from seaborn import heatmap
from data_manipulation import X_test, y_test
import datetime

# Set the model name
model_name = 'trained_model1'

# Create the main folder where all results will be saved
main_folder = './results_plots/'

# Create the main directory if it doesn't exist
os.makedirs(main_folder, exist_ok=True)

# Get the current timestamp to ensure unique folder names
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create the folder for this specific model and execution
model_folder = os.path.join(main_folder, f'{model_name}_{timestamp}')
os.makedirs(model_folder, exist_ok=True)

# Load the trained model
model = load_model(f'models/{model_name}.keras')

# Load training history
with open(f'history/history_{model_name}', 'rb') as f:
    history = pickle.load(f)

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
plt.savefig(f'{model_folder}/training_validation_curves.png')

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
plt.savefig(f'{model_folder}/confusion_matrix.png')

"""PREDICTIONS ON TEST DATA"""
y_pred = model.predict(X_test)
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

# Plot predicted vs actual for a few examples
fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title=f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}")

# Save the predictions plot
plt.savefig(f'{model_folder}/predictions_on_test_data.png')

print(f"Plots saved successfully in folder: {model_folder}")
