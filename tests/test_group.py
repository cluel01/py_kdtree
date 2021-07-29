import numpy as np 
from py_kdtree.kdtree import KDTree
np.random.seed(42)

X = np.random.randn(1000,3).astype(np.float32)

tree = KDTree(leaf_size=20,path="/home/cluelf/py_kdtree/run",dtype="float32",chunksize="leaf")

tree.fit(X)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.5,.1,.1]))
print(len(inds))
print(len(pts))
#print(inds)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.5,0.1,0.1]))

print(len(inds))
print(len(pts))
print(tree.depth)