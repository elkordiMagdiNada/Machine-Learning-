import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept

    def add_intercept(self, X):
        intercept = np.ones((len(X), 1))
        return np.concatenate((intercept, X), axis=1)

    def normalize_features(self, X):
        X = np.array(X)
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / sigma

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X = self.normalize_features(X)

        if self.fit_intercept:
            X = self.add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        X = np.array(X)

        X = self.normalize_features(X)

        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        X = np.array(X)

        return self.predict_prob(X) >= threshold

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
# Load the data: training set from a space-separated file
#data = np.loadtxt('emails.csv')
data = pd.read_csv('emails.csv', skiprows=[0])

# Split the data into features and labels
X = (data.iloc[:, 1 :-1]).to_numpy()
Y = (data.iloc[:, -1]).to_numpy()



model = LogisticRegression(0.1, 3000)


# Create a 5-fold cross-validation object
kfold = KFold(n_splits=5)

# Perform 5-fold cross-validation
fold_num = 1
sum_acc =0
for train_index, test_index in kfold.split(X):
    #print('Fold', fold_num)
    #print('Training samples:', train_index)
    #print('Test samples:', test_index)
    X_train = []
    X_test =[]
    y_train =[]
    y_test =[]
    for i in train_index:
        X_train.append(X[i])
        y_train.append(Y[i])
    for i in test_index:
        X_test.append(X[i])
        y_test.append(Y[i])
    #X_train, X_test = [X[i-1] for i in train_index], [X[i-1] for i in test_index]
    #y_train, y_test = [Y[i-1] for i in train_index], [Y[i-1] for i in test_index]
    model.fit(X_train, y_train)
    # accuracy = knn.score(X_test, y_test)
    # sum_acc += accuracy
    #print('Accuracy:', accuracy)
    y_test_predict = model.predict(X_test)
    binary_arr = y_test_predict.astype(int)
    accuracy = accuracy_score(y_test, binary_arr)
    recall = recall_score(y_test, binary_arr,zero_division=1)
    precision = precision_score(y_test, binary_arr,zero_division=1)

    print("Accuracy: {:.2f}".format(accuracy))
    print("Recall: {:.2f}".format(recall))
    print("Precision: {:.2f}".format(precision))
    fold_num += 1
