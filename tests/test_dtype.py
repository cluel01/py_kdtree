import numpy as np 
from py_kdtree.kdtree import KDTree

X = np.random.randn(100,2).astype("float32")

tree = KDTree(leaf_size=10,path="./run",dtype="float64")

tree.fit(X)