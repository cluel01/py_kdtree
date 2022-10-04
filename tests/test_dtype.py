import numpy as np 
from py_kdtree.kdtree import KDTree

X_org = np.random.randn(100,2)
X = X_org.astype("float32")

tree = KDTree(leaf_size=10,path="./run",dtype="float64")

tree.fit(X)


X_test = np.random.randn(2)

print(tree.query_point_cy(X_test))

tree2 = KDTree(leaf_size=10,path="./run2",dtype="float64")
tree2.fit(X_org)

print(tree2.query_point_cy(X_test))