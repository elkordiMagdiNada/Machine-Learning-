import math

import numpy as np

np.random.seed(5)
# Generate random data with two values
X = np.random.random(size=100)
# Add Gaussian noise to each value
noise = np.random.normal(0, 1, size=100)
values_with_noise = X + noise

# Write the data to a file
with open("sin_test_100.txt", "w") as f:
    for i in range(100):
        f.write("{:.6f} {:.6f}\n".format(values_with_noise[i], np.sin(X[i])))