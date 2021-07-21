import numpy as np 
from kdtree import KDTree
np.random.seed(42)

X = np.random.randn(100000,3)

tree = KDTree(leaf_size=100,path="/home/cluelf/py_kdtree/run")

if tree._root is None:
    tree.fit(X)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.5,.1,.1]))
print(len(inds))
print(len(pts))
print(inds)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.5,0.1,0.1]))

print(len(inds))
print(len(pts))