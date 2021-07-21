import numpy as np 
from py_kdtree.kdtree import KDTree
np.random.seed(42)

X = np.random.randn(1000,3)

tree = KDTree(leaf_size=100,path="/home/cluelf/py_kdtree/run")

tree.fit(X)
