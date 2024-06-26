import numpy as np 
from py_kdtree.kdtree import KDTree
np.random.seed(42)

X = np.random.randn(10000000,3).astype(np.float32)

tree = KDTree(leaf_size=40,path="/home/cluelf/py_kdtree/run",dtype="float32")

tree.fit(X)

inds,pts,nleaves,time = tree.query_box(np.array([0,0,0]),np.array([0.5,.1,.1]))

print(len(inds))
print(len(pts))
#print(inds)
print("nleaves: ",nleaves)



