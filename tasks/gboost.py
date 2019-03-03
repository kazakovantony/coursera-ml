from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from sklearn import datasets, cross_validation, tree, metrics, ensemble, linear_model
from matplotlib import pyplot as plt
import seaborn
import numpy as np

boston = datasets.load_boston()
X = boston.data
Y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=51)

print(X.shape, Y.shape)

base_algorithms_list = []
coefficients_list = []


def gbm_predict(X):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)])
            for x in X]


def get_grad():
    return [y - a for a, y in zip(gbm_predict(X_train), y_train)]
# or more simple return y_train - gbm_predict(X_train)


for i in np.arange(0, 50):
    # create new algorithm
    rg = DecisionTreeRegressor(random_state=42, max_depth=5)
    # fit algo in train dataset and new target
    rg.fit(X_train, get_grad())
    # append results
    base_algorithms_list.append(rg)
    coefficients_list.append(0.9)

pred = gbm_predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))


xbg = XGBRegressor(n_estimators=50, max_depth=5)
xbg.fit(X_train, y_train)
pred = xbg.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))


def test_xbg():
    plt.figure(figsize=(15, 8))

    trees = [50, 100, 200, 300, 400, 500, 1000]
    errors = []
    for tree in trees:
        errors.append(
            -cross_val_score(XGBRegressor(n_estimators=tree), X, Y,  scoring='neg_mean_squared_error').mean()
        )
    plt.subplot(121)
    plt.plot(trees, errors)
    plt.xlabel("trees")
    plt.ylabel("error")
    plt.title("number trees")

    depth = [2, 4, 6, 8, 20]
    errors = []
    for d in depth:
        errors.append(
            -cross_val_score(XGBRegressor(max_depth=d), X, Y,  scoring='neg_mean_squared_error').mean()
        )
    plt.subplot(122)
    plt.plot(depth, errors)
    plt.xlabel("depth")
    plt.ylabel("error")
    plt.title("tree depth")
    plt.show()


print(test_xbg())

number_trees = np.arange(5, 1000, 5)
train_scores = []
test_scores = []

for tree in number_trees:
    print(tree),
    clf = ensemble.GradientBoostingRegressor(n_estimators=tree)
    clf.fit(X_train, y_train)
    train_scores.append(metrics.mean_squared_error(y_train, clf.predict(X_train)))
    test_scores.append(metrics.mean_squared_error(y_test, clf.predict(X_test)))


plt.plot(number_trees, train_scores)
plt.plot(number_trees, test_scores)

tree_depth = np.arange(1, 50, 1)
train_scores = []
test_scores = []

for depth in tree_depth:
    print(depth),
    clf = ensemble.GradientBoostingRegressor(max_depth=depth)
    clf.fit(X_train, y_train)
    train_scores.append(metrics.mean_squared_error(y_train, clf.predict(X_train)))
    test_scores.append(metrics.mean_squared_error(y_test, clf.predict(X_test)))


plt.plot(tree_depth, train_scores)
plt.plot(tree_depth, test_scores)


clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

print(metrics.mean_squared_error(y_test, clf.predict(X_test)) ** 0.5)

plt.scatter(y_test, clf.predict(X_test), color = 'blue')
plt.scatter(y_train, clf.predict(X_train), color = 'red')
plt.xlabel('target')
plt.ylabel('prediction')
