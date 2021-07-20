import numpy as np 
from kdtree import KDTree
np.random.seed(42)

X = np.random.randn(10000000,3)

tree = KDTree(X,leaf_size=500000)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.1,0.1,0.1]))
print(len(inds))
print(inds)
