from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target
iris

clf = DecisionTreeClassifier(criterion="gini", random_state=12)
model = clf.fit(x,y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)


import dtreeviz

viz = dtreeviz.model(clf, x, y,
                     target_name="target",
                     feature_names=iris.feature_names,
                     class_names=list(iris.target_names))

viz.view()

from supertree import SuperTree

supertree = SuperTree(model, x,y, feature_names= iris.feature_names, target_names=iris.target_names)
supertree.show_tree()

