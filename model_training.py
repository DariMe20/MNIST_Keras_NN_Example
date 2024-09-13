import keras
import pickle

from keras.models import load_model
from data_augmentation import train_gen, test_gen
from data_manipulation import X_train, X_test

model = load_model("models/model1.keras")

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

# Save the trained model
model.save("models/trained_model1.keras")

# Save the training history
with open('history/history_trained_model1', 'wb') as f:
    pickle.dump(history.history, f)

print("Model and history saved successfully!")
