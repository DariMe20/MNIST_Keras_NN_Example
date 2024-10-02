from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Each layer is added one after the other => Sequential arhitecture
model = Sequential()

model.add(Conv2D(filters=2, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# Save the initial model
model.save("models/model_1layer_network.h5")
