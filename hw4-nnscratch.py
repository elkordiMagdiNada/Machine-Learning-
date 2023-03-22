import numpy as np
import matplotlib.pyplot as plt
import random

# Load MNIST dataset
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten images
train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

# Define neural network architecture
input_size = 784
hidden_size1 = 300
hidden_size2 = 200
output_size = 10
train_loss_history = []
test_loss_history = []

# Initialize weights
w1 = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size)
b1 = np.zeros(hidden_size1)
w2 = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1)
b2 = np.zeros(hidden_size2)
w3 = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2)
b3 = np.zeros(output_size)

# Define activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of activation function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define loss function
def cross_entropy_loss(y_hat, y):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Define learning rate and number of epochs
learning_rate = 0.1
num_epochs = 10
batch_size = 32

num_batches = int(np.ceil(train_images.shape[0] / batch_size))
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in range(num_batches):
        # Get batch
        batch_start = batch * batch_size
        batch_end = min(batch_start + batch_size, train_images.shape[0])
        batch_images = train_images[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]

        # Forward pass
        a1 = np.dot(batch_images, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y_hat = sigmoid(a3)

        # Compute loss and backpropagate
        y = np.zeros((batch_size, output_size))
        y[np.arange(batch_size), batch_labels] = 1.0
        loss = cross_entropy_loss(y_hat, y)
        total_loss += loss
        delta_a3 = y_hat - y
        delta_w3 = np.dot(z2.T, delta_a3) / batch_size
        delta_b3 = np.mean(delta_a3, axis=0)
        delta_a2 = np.dot(delta_a3, w3.T) * sigmoid_derivative(a2)
        delta_w2 = np.dot(z1.T, delta_a2) / batch_size
        delta_b2 = np.mean(delta_a2, axis=0)
        delta_a1 = np.dot(delta_a2, w2.T) * sigmoid_derivative(a1)
        delta_w1 = np.dot(batch_images.T, delta_a1) / batch_size
        delta_b1 = np.mean(delta_a1, axis=0)
        # Update weights and biases
        w3 -= learning_rate * delta_w3
        b3 -= learning_rate * delta_b3
        w2 -= learning_rate * delta_w2
        b2 -= learning_rate * delta_b2
        w1 -= learning_rate * delta_w1
        b1 -= learning_rate * delta_b1
    avg_loss = total_loss / num_batches

    # Compute accuracy on test set
    num_correct = 0
    for i in range(test_images.shape[0]):
        a1 = np.dot(test_images[i], w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y_hat = sigmoid(a3)

        test_loss = cross_entropy_loss(y_hat, y)
        test_loss_history.append(test_loss)
        if np.argmax(y_hat) == test_labels[i]:
            num_correct += 1
    accuracy = num_correct / test_images.shape[0]
    train_loss_history.append(accuracy)
    print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, total_loss/train_images.shape[0], accuracy))

plt.plot(train_loss_history)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()