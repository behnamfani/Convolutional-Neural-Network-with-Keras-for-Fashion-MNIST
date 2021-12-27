from keras.datasets import fashion_mnist
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

keras.backend.set_image_data_format('channels_first')

# Reading the data and dividing it to train, validation, and test sets.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("x_train shape:", x_train.shape, " y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape, " y_test shape:", y_test.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]


w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], 1, w, h)
x_valid = x_valid.reshape(x_valid.shape[0], 1, w, h)
x_test = x_test.reshape(x_test.shape[0], 1, w, h)
y_train = keras.utils.to_categorical(y_train, 10)
y_valid = keras.utils.to_categorical(y_valid, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Defining and constructing the model
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', data_format='channels_first',
                              input_shape=(1, 28, 28)))
model.add(keras.layers.MaxPooling2D(pool_size=2, data_format='channels_first'))
model.add(keras.layers.Dropout(0.3))
model.add(
    keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu', data_format='channels_first'))
model.add(keras.layers.MaxPooling2D(pool_size=2, data_format='channels_first'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model on train and validation sets and saving the weights
checkpointer = keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_valid, y_valid), callbacks=[checkpointer])

# Plotting history of model's accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
# Plotting history of model's loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model (categorical_crossentropy) Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Checking the model performance on test data 
model.load_weights('model.weights.best.hdf5')
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])
