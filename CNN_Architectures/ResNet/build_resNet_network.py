from keras.layers import Input, Conv2D, BatchNormalization, Add, Activation, Flatten, Dense
from keras.models import Model


# Function to create a residual block
def residual_block(x, filters):
    shortcut = x  # Save the input for the shortcut
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x)  # First convolution

    x = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x)  # Second convolution

    x = Add()([x, shortcut])  # Add the shortcut connection
    return x


# Build the ResNet model using the Functional API
input_layer = Input(shape=(28, 28, 1))  # Input layer

# Initial convolutional layer
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

# Add residual blocks
for _ in range(1):
    x = residual_block(x, 32)

# Final layers
x = Flatten()(x)  # Flatten the output for dense layers
x = Dense(128, activation='relu')(x)  # Fully connected layer
output_layer = Dense(10, activation='softmax')(x)  # Output layer for 10 classes

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

# Save the model
model.save("models/ResNet_model1.h5")
