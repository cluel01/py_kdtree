import numpy as np 
from kdtree import KDTree
np.random.seed(42)

X = np.random.randn(10000000,3)

tree = KDTree(X,leaf_size=10000,path="/home/cluelf/py-kdtree/run")

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.1,.1,.1]))
print(len(inds))
print(len(pts))
print(inds)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.1,0.1,0.1]))

print(len(inds))
print(len(pts))