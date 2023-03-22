# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tensorflow.keras import initializers

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Import required libraries


    # Set random seed for reproducibility
    np.random.seed(0)

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape input data to a 1D vector and scale to 0-1 range
    X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

    # Convert output data to one-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Define the neural network architecture
    model = Sequential()
    model.add(Dense(300, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model on test data
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    # Plot the learning curve
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
