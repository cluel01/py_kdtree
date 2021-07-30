import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.treeset import KDTreeSet
np.random.seed(42)

X = np.random.randn(1000,7).astype(np.float32)

idxs = [[0,1,2],[3,4,5,6],[0,1,3]]


ens = KDTreeSet(idxs,leaf_size=20,path="/home/cluelf/py_kdtree/run",dtype="float32",verbose=True)

ens.fit(X)


inds,pts = ens.query(np.array([0,0,0]),np.array([0.5,.1,.1]),idxs[0])
print(len(inds))
print(len(pts))
#print(inds)

inds,pts = ens.query(np.array([0,0,0,0]),np.array([0.5,0.1,0.1,0.1]),idxs[1])

print(len(inds))
print(len(pts))

inds,pts = ens.multi_query(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.1],[0.1,0.1,0.2]]),[idxs[i] for i in [0,2]])
print(len(inds))