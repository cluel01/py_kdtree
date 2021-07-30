import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.treeset import KDTreeSet
np.random.seed(42)
import os

x1 = np.random.randn(100,128).astype(dtype="float32")
x2 = np.random.randn(200,128).astype(dtype="float32")
np.save("/home/cluelf/py_kdtree/run/x1.npy",x1)
np.save("/home/cluelf/py_kdtree/run/x2.npy",x2)

x_files = [i for i in os.listdir("/home/cluelf/py_kdtree/run/") if i.endswith(".npy")]

idxs= [[79, 86, 97], [12, 67, 87], [107, 90, 97], [1, 125, 55], [102, 122, 41], [40, 72, 82], [100, 105, 42], [23, 45, 86], [118, 16, 28]]


ens = KDTreeSet(idxs,leaf_size=20,path="/home/cluelf/py_kdtree/run",dtype="float32")

ens.fit_seq(x_files,n_cached=3)
#ens.fit(np.vstack([x1,x2]))


inds,pts = ens.query(np.array([0,0,0]),np.array([0.5,.1,.1]),idxs[0])
print(len(inds))
print(len(pts))
#print(inds)

print(len(inds))
print(len(pts))

idxs = [[0,1,4]]


ens = KDTreeSet(idxs,leaf_size=20,path="/home/cluelf/py_kdtree/run",dtype="float32")

ens.fit_seq(x_files,n_cached=3)