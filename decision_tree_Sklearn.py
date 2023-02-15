from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from graphviz import Source
import numpy as np

# Load the data from a space-separated file
data = np.loadtxt('D8192.txt')

# Split the data into features and labels
X = data[:, :-1]
y = data[:, -1]



# Build a decision tree classifier with gain ratio as the splitting criterion
clf = DecisionTreeClassifier(criterion='entropy', splitter='best')

# Fit the classifier to the training data
clf.fit(X, y)
print (f"num nodes: {clf.tree_.node_count}")

test_data = np.loadtxt('Dbig.txt')
X_test = test_data[:, :-1]
y_test = test_data[:, -1]
y_pred = clf.predict(X_test)

test_set_error = sum(y_pred != y_test) / len(y_test)
print (f"test_set_error: {test_set_error}")


#text_representation = tree.export_text(clf)
#print(text_representation)

