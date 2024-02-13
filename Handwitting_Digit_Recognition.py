import os
import cv2
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.layers import Dense
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

###########################################################################

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(60000, 28, 28, 1)
test_x = test_x.reshape(10000, 28, 28, 1)
train_x = train_x.astype('float32') / 255.0  # Normalize pixel values to between 0 and 1
test_x = test_x.astype('float32') / 255.0
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

###########################################################################

model = Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
print("_________________________________________________________________")
print("Summary of the neural network used")
print(model.summary())

#############################################################################

key_pressed = ord(input(f"Do you want to train a neural network[y/n]?:"))
key_pressed1 = ord(input(f"Do you want to test the neural network[y/n]?:"))
key_pressed2 = ord(input(f"Do you want to predict the numbers[y/n]?:"))

#############################################################################

if key_pressed == ord("y") or key_pressed == ord("Y"):

    history = model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1, validation_data=(test_x, test_y))

    model.save("mnist_model.h5")

    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()
if key_pressed1 == ord("y") or key_pressed1 == ord("Y"):
    ###########################################################################

    model.load_weights("mnist_model.h5")

    test_loss, test_accuracy = model.evaluate(test_x, test_y, batch_size=1, verbose=1)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

if key_pressed2 == ord("y") or key_pressed2 == ord("Y"):
    ###########################################################################

    model.load_weights("mnist_model.h5")

    image_number = 1
    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error!")

        finally:
            image_number += 1
