import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score


# Load the data: training set from a space-separated file
#data = np.loadtxt('emails.csv')
data = pd.read_csv('emails.csv', skiprows=[0])

# Split the data into features and labels
X = (data.iloc[:, 1 :-1]).to_numpy()
Y = (data.iloc[:, -1]).to_numpy()




knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

# Create a 5-fold cross-validation object
kfold = KFold(n_splits=5)

# Perform 5-fold cross-validation
fold_num = 1
for train_index, test_index in kfold.split(X):
    print('Fold', fold_num)
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
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print('Accuracy:', accuracy)
    y_test_predict = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_predict)
    recall = recall_score(y_test, y_test_predict)
    precision = precision_score(y_test, y_test_predict)

    # Print the results
    print("Accuracy: {:.2f}".format(accuracy))
    print("Recall: {:.2f}".format(recall))
    print("Precision: {:.2f}".format(precision))
    fold_num += 1
