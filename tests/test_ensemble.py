import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.ensemble import KDTreeEnsemble
np.random.seed(42)

X = np.random.randn(1000,6).astype(np.float32)

idxs = [[0,1,2],[3,4,5]]


ens = KDTreeEnsemble(idxs,leaf_size=20,path="/home/cluelf/py_kdtree/run",dtype="float32",chunksize="leaf")

ens.fit(X)

'''
inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.5,.1,.1]))
print(len(inds))
print(len(pts))
#print(inds)

inds,pts = tree.query_box(np.array([0,0,0]),np.array([0.5,0.1,0.1]))

print(len(inds))
print(len(pts))
print(tree.depth)
'''