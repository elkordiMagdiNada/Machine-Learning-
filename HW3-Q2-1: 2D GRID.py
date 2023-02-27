import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the data: training set from a space-separated file
data = np.loadtxt('D2z.txt')

# Split the data into features and labels
X_train= data[:, :-1]
Y_train = data[:, -1]

# create an empty list to hold the grid
x_test = []

# loop through x and y coordinates
for x in range(-20, 20, 1):

    for y in range(-20, 20, 1):
        # calculate the x and y values
        # add the x and y values to the current \
        row = []
        row.append(x*.1)
        row.append(y * .1)
    # add the current row to the grid
        x_test.append(row)


clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(X_train, Y_train)
y_pred = clf.predict(x_test)
#
# # Plot the data as a line
# ax.plot(X_train, y_pred)
#
# # Set the axis labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('1NN 2D Grid')



# Create a scatter plot
fig, ax = plt.subplots()
for i, label in enumerate(set(Y_train)):
    x = [X_train[j][0] for j in range(len(X_train)) if Y_train[j] == label]
    y = [X_train[j][1] for j in range(len(X_train)) if Y_train[j] == label]
    ax.scatter(x, y, label=label)


for i, label in enumerate(set(y_pred)):
    x = [x_test[j][0] for j in range(len(x_test)) if y_pred[j] == label]
    y = [x_test[j][1] for j in range(len(x_test)) if y_pred[j] == label]
    if label == 0 :
        ax.scatter(x, y, label='.')
    else :
        ax.scatter(x, y, label='o')

# Add legend and labels to the plot
ax.legend()
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Scatter Plot 1NN')


# Show the plot
plt.show()
