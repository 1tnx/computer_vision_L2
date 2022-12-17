import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# add inversed images to training set
x_train = np.concatenate((x_train, 255 - x_train), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)
x_test = np.concatenate((x_test, 255 - x_test), axis=0)
y_test = np.concatenate((y_test, y_test), axis=0)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save('model.h5')
