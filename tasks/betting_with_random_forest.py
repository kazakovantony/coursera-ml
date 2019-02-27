
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

digits = load_digits()
X = digits.data
Y = digits.target

print(X.shape, Y.shape)


def plot_number_by_data(img_data, label):
    plt.figure(1, figsize=(3, 3))
    plt.imshow(img_data.reshape((8,8)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    plt.title(f"label is {label}")
    plt.show()


def plot_number_by_index(ind):
    dt = X[ind]
    label = Y[ind]
    plot_number_by_data(dt, label)


plot_number_by_index(9)


# just evaluate mean 10-fold cross validation score.
def fit_estimator(estimator):
    return cross_val_score(estimator, X, Y, cv=10).mean()


# plot digits where classifier made mistake.
def plot_invalid_labels(estimator):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=51)
    estimator.fit(X_train, y_train)
    predict = estimator.predict(X_test)

    fig=plt.figure(figsize=(15, 10))
    columns = 5
    rows = 4
    j = 1
    for i in np.arange(len(predict)):
        predicted = predict[i]
        actual = y_test[i]
        if predicted != actual:
            if j <= rows*columns:
                img = X_test[i].reshape((8,8))
                fig.add_subplot(rows, columns, j)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                fig.tight_layout()
                plt.title(f"label is {actual} pr {predicted}")
                j += 1
    plt.show()


d_tree1 = DecisionTreeClassifier(random_state=17)
print(fit_estimator(d_tree1))
plot_invalid_labels(d_tree1)

d_tree = DecisionTreeClassifier(random_state=37)
bagging = BaggingClassifier(base_estimator=d_tree, random_state=11, n_estimators=100)
print(fit_estimator(bagging))

d_tree = DecisionTreeClassifier(random_state=37)
bagging = BaggingClassifier(base_estimator=d_tree, random_state=11, n_estimators=100, max_features=int(np.sqrt(X.shape[1])))
print(fit_estimator(bagging))

d_tree = DecisionTreeClassifier(random_state=37, max_features=int(np.sqrt(X.shape[1])))
bagging = BaggingClassifier(base_estimator=d_tree, random_state=11, n_estimators=100)
print(fit_estimator(bagging))


def plot_rf_trees_score():
    trees = [100, 200, 300, 400, 500, 1000]
    results = []
    for tree in trees:
        rf = RandomForestClassifier(n_estimators=tree)
        results.append(fit_estimator(rf))
    plt.figure(figsize=(15, 8))
    plt.plot(trees, results)
    plt.xlabel("n-trees")
    plt.ylabel("score")
    plt.title("Trees score dependencies")
    plt.show()


plot_rf_trees_score()


def plot_rf_trees_max_features():
    d = X.shape[1]
    features = [2, int(np.sqrt(d)), int(d/3), d]
    results = []
    for f in features:
        rf = RandomForestClassifier(n_estimators=400, random_state=101, max_features=f)
        results.append(fit_estimator(rf))
    plt.figure(figsize=(15, 8))
    plt.plot(features, results, 'o')
    plt.xlabel("features")
    plt.ylabel("score")
    plt.title("Trees feature dependencies")
    plt.show()


plot_rf_trees_max_features()


def plot_rf_tree_depth():
    d = X.shape[1]
    depth = [2, 4, 6, 8]
    results = []
    for d in depth:
        rf = RandomForestClassifier(n_estimators=400, random_state=101, max_depth=d)
        results.append(fit_estimator(rf))
    plt.figure(figsize=(15, 8))
    plt.plot(depth, results, 'o')
    plt.xlabel("features")
    plt.ylabel("score")
    plt.title("Trees depth dependencies")
    plt.show()


plot_rf_tree_depth()

# 2 3 4 7
