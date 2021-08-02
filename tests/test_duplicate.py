import numpy as np 
from py_kdtree.kdtree import KDTree
np.random.seed(42)

X = np.random.randint(0,100,(1000000,3)).astype(np.float32)
print(len(np.unique(X,axis=0)))

tree = KDTree(leaf_size=20000,path="/home/cluelf/py_kdtree/run",dtype="float32")

tree.fit(X)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([10,10,10]))
print(len(inds))
print(len(np.unique(inds)))
#print(inds)




