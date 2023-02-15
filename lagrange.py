from scipy.interpolate import lagrange
import numpy as np

data_train = np.loadtxt('sin_training.txt')
x_train = data_train[:, 0]
y_train = data_train[:, -1]

poly = lagrange(x_train, y_train)

data_test = np.loadtxt('sin_test_100.txt')
x_test = data_test[:, 0]
y_test = data_test[:, -1]

y_pred = poly(x_test)

error = np.sqrt(np.mean((y_test - y_pred)**2))


print("Test error):", error)