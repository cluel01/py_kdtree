import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.treeset import KDTreeSet
np.random.seed(42)
import os

x1 = np.random.randn(100000,7).astype(dtype="float32")
x2 = np.random.randn(200000,7).astype(dtype="float32")
np.save("/home/cluelf/py_kdtree/run/x1.npy",x1)
np.save("/home/cluelf/py_kdtree/run/x2.npy",x2)

x_files = [i for i in os.listdir("/home/cluelf/py_kdtree/run/") if i.endswith(".npy")]

idxs = [[0,1,2],[3,4,5,6],[0,1,3]]


ens = KDTreeSet(idxs,leaf_size=1000,path="/home/cluelf/py_kdtree/run",dtype="float32")

ens.fit_seq(x_files,n_cached=3)
#ens.fit(np.vstack([x1,x2]))


inds,pts = ens.query(np.array([0,0,0]),np.array([0.5,.1,.1]),idxs[0])
print(len(inds))
print(len(pts))
#print(inds)

inds,pts = ens.query(np.array([0,0,0,0]),np.array([0.5,0.1,0.1,0.1]),idxs[1])

print(len(inds))
print(len(pts))

inds,pts = ens.multi_query(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.3],[0.5,0.5,0.4]]),[idxs[i] for i in [0,2]])
print(len(inds))
print(len(np.unique(inds)))


idxs = [[0,1,4]]



ens = KDTreeSet(idxs,leaf_size=2000,path="/home/cluelf/py_kdtree/run",dtype="float32")

ens.fit_seq(x_files,n_cached=3)