import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from scipy import interp
from scipy.interpolate import CubicSpline
from sklearn.metrics import accuracy_score, recall_score, precision_score


# Load the data: training set from a space-separated file
#data = np.loadtxt('emails.csv')
data = pd.read_csv('emails.csv', skiprows=[0])

# Split the data into features and labels
X = (data.iloc[:, 1 :-1]).to_numpy()
Y = (data.iloc[:, -1]).to_numpy()




knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
logreg = LogisticRegression()
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
    knn.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    sum_acc += accuracy
    #print('Accuracy:', accuracy)
    # y_test_predict = knn.predict(X_test)
    # accuracy = accuracy_score(y_test, y_test_predict)
    # recall = recall_score(y_test, y_test_predict)
    # precision = precision_score(y_test, y_test_predict)

    # Print the results
    # print("Accuracy: {:.2f}".format(accuracy))
    # print("Recall: {:.2f}".format(recall))
    # print("Precision: {:.2f}".format(precision))
    fold_num += 1
    train_probs = knn.predict_proba(X_train)[:,1]
    test_probs = knn.predict_proba(X_test)[:,1]
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probs)
    test_fpr, test_tpr, _ = roc_curve(y_test, test_probs)

    log_train_probs = logreg.predict_proba(X_train)[:,1]
    log_test_probs = logreg.predict_proba(X_test)[:,1]
    log_train_fpr, log_train_tpr, _ = roc_curve(y_train, log_train_probs)
    log_test_fpr, log_test_tpr, _ = roc_curve(y_test, log_test_probs)

    # Compute the AUC for the training and test sets
    train_auc = auc(train_fpr, train_tpr)
    test_auc = auc(test_fpr, test_tpr)
    log_test_auc = auc(log_test_fpr, log_test_tpr)

    mean_fpr = np.linspace(0, 1, 100)
    tpr_interp = interp(mean_fpr, test_fpr, test_tpr)
    roc_auc_interp = auc(mean_fpr, tpr_interp)
    # Smooth the ROC curve using cubic spline interpolation
    # cs = CubicSpline(test_fpr, test_tpr)
    # smooth_fpr = np.linspace(0, 1, 100)
    # smooth_tpr = cs(smooth_fpr)
    # Plot the ROC curves for the training and test sets
    #plt.plot(train_fpr, train_tpr, label="Training set (AUC = {:.2f})".format(train_auc))
    plt.plot(test_fpr, test_tpr, label="KNNClassifier (AUC = {:.5f})".format(test_auc))
    # plt.plot(mean_fpr, tpr_interp, label="SmoothKNNClassifier (AUC = {:.5f})".format(roc_auc_interp))

    plt.plot(log_test_fpr, log_test_tpr, label="LogReglassifier (AUC = {:.5f})".format(test_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    break

