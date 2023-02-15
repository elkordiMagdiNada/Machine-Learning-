import csv
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature_index=None, split_value=None, then_child=None, else_child=None, leaf_label=None):
        self.feature_index = feature_index
        self.split_value = split_value
        self.then_child = then_child
        self.else_child = else_child
        self.leaf_label = leaf_label

    def is_leaf_node(self):
        return self.leaf_label is not None

def predict (node, data):
    if node.is_leaf_node():
        return node.leaf_label
    if data[node.feature_index]< float(node.split_value) :
        return predict(node.then_child,data)
    return predict(node.else_child, data)


def predict_arr (node, data_points):
    predictions = list()
    for row in data_points:
        result=  predict(node,row)
        predictions.append(result)
    return predictions
def visualize_decision_tree(node, depth=0):
    indent = "    " * depth
    if node.is_leaf_node():
        print(f"{indent}Class {node.leaf_label} Depth: {depth}")
        return 1
    else:

        print(f"{indent}Depth {depth}.1 :: x[{node.feature_index}] < {node.split_value} :")
        num = visualize_decision_tree(node.then_child, depth + 1)
        print(f"{indent}Depth {depth}.2 ::else:")
        num += visualize_decision_tree(node.else_child, depth + 1)
        return num


def read_input(input_file):
    return list(csv.reader(open(input_file), delimiter=" "))


def find_candidate_splits(input_data_to_candidate):
    sorted_data_feature_0 = sorted(input_data_to_candidate, key=lambda row: (row[0]))

    candidate_splits_0 = []
    candidate_splits_1 = []
    # Sort on feature 0
    previous_row = sorted_data_feature_0[0]
    index = 1
    while index < len(sorted_data_feature_0):
        if sorted_data_feature_0[index][2] != previous_row[2]:
            candidate_splits_0.append({'feature_index': 0, 'split_value': sorted_data_feature_0[index][0]})
            previous_row = sorted_data_feature_0[index]
        index += 1

    # Sort on feature 1
    sorted_data_feature_1 = sorted(input_data_to_candidate, key=lambda row: (row[1]))
    previous_row = sorted_data_feature_1[0]

    index = 1
    while index < len(sorted_data_feature_1):
        if sorted_data_feature_1[index][2] != previous_row[2]:
            candidate_splits_1.append({'feature_index': 1, 'split_value': sorted_data_feature_1[index][1]})
            previous_row = sorted_data_feature_1[index]
        index += 1
    return candidate_splits_0, candidate_splits_1, sorted_data_feature_0, sorted_data_feature_1


def make_split_on_candidate(candidate_split, input_data_to_split):
    then_branch, else_branch = list(), list()
    for row in input_data_to_split:
        if row[candidate_split['feature_index']] < candidate_split['split_value']:
            then_branch.append(row)
        else:
            else_branch.append(row)
    return then_branch, else_branch


def calc_entropy(column):
    elements, counts = np.unique(column, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def get_gain_ratio(candidate_split, feature_index, input_data_sorted_on_feature):
    labels_sorted = [item[2] for item in input_data_sorted_on_feature]
    n = len(labels_sorted)
    then_features_sorted = [d for d in input_data_sorted_on_feature if
                            d[feature_index] < candidate_split['split_value']]
    else_features_sorted = [d for d in input_data_sorted_on_feature if
                            d[feature_index] >= candidate_split['split_value']]
    left_labels = [labels_sorted[i] for i, d in enumerate(input_data_sorted_on_feature) if
                   d[feature_index] < candidate_split['split_value']]
    right_labels = [labels_sorted[i] for i, d in enumerate(input_data_sorted_on_feature) if
                    d[feature_index] >= candidate_split['split_value']]
    info_gain = calc_entropy(labels_sorted) - (len(left_labels) / n) * calc_entropy(left_labels) - (
                len(right_labels) / n) * calc_entropy(
        right_labels)
    #print(f"----Information gain:{info_gain}")
    # This validates the termination condition 3 when the entropy of a split is zero, that the split results in all data
    # In a single bucket, this returns the gain ratio as zero
    if len(then_features_sorted) == 0:
        return 0
    if len(else_features_sorted) == 0:
        return 0
    left_ratio = len(then_features_sorted) / n
    right_ratio = len(else_features_sorted) / n
    split_info = -(left_ratio * np.log2(left_ratio) + right_ratio * np.log2(right_ratio))
    #print(f"----Gain ratio:{info_gain / split_info}")
    return info_gain / split_info


def build_decision_tree(input_data_to_build):
    # Stopping criteria 1: node is empty and predict a class of 1
    if len(input_data_to_build) == 0:
        return Node(leaf_label=1)

    candidate_splits_0, candidate_splits_1, sorted_data_feature_0, sorted_data_feature_1 = find_candidate_splits(
        input_data_to_build)
    # Checking splits on feature 0
    gain_ratio = 0
    best_split = []

    for candidate_split_0 in candidate_splits_0:
        #print(f"candidate_split on feature x0:{candidate_split_0['split_value']}")
        gain_ratio_new = get_gain_ratio(candidate_split_0, 0, sorted_data_feature_0)
        if gain_ratio_new > gain_ratio:
            gain_ratio = gain_ratio_new
            best_split = candidate_split_0
            then_branch, else_branch = make_split_on_candidate(candidate_split_0, sorted_data_feature_0)

    # Checking splits on feature 1
    for candidate_split_1 in candidate_splits_1:
        #print(f"candidate_split on feature x1:{candidate_split_1['split_value']}")
        gain_ratio_new = get_gain_ratio(candidate_split_1, 1, sorted_data_feature_1)
        if gain_ratio_new > gain_ratio:
            gain_ratio = gain_ratio_new
            best_split = candidate_split_1
            then_branch, else_branch = make_split_on_candidate(candidate_split_1, sorted_data_feature_1)

    # Stopping criteria 2 all splits have zero gain ratio
    if gain_ratio == 0:
        # get the majority of the labels
        labels = [item[2] for item in input_data_to_build]
        leaf_label = np.argmax(np.bincount(labels))
        return Node(leaf_label=leaf_label)
    #print("==============================================================")
    then_node_child = build_decision_tree(then_branch)
    else_node_child = build_decision_tree(else_branch)
    return Node(best_split['feature_index'], best_split['split_value'], then_node_child, else_node_child)


input_data = read_input("D8192.txt")
decision_tree = build_decision_tree(input_data)
#num = visualize_decision_tree(decision_tree, 0)
#print (f"Num nodes {num}")


data = np.loadtxt('Dbig.txt')
X = data[:, :-1]
Y = data[:, -1]
Z = predict_arr(decision_tree, X)
error = sum(1 for i, j in zip(Z, Y) if i != j)/len(Y)
print(f"Error {error}")

# data = np.loadtxt('D2048.txt')
# X = data[:, :-1]
# y = data[:, -1]
#
# # Define a grid of points in the feature space
# x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
# y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
#                      np.linspace(y_min, y_max, 10))
# Z = predict_arr(decision_tree, np.c_[xx.ravel(), yy.ravel()])
# Z_np = np.array(Z)
# b = Z_np.reshape(xx.shape)
# # Plot the decision boundary
# plt.contourf(xx, yy, b, alpha=0.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()
