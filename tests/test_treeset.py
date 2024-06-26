import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.treeset import KDTreeSet
np.random.seed(42)

X = np.random.randn(100000,7).astype(np.float32)

idxs = [[0,1,2],[3,4,5,6],[0,1,3]]


ens = KDTreeSet(idxs,leaf_size=1000,path="/home/cluelf/py_kdtree/run",dtype="float32",group_prefix="",verbose=True)

ens.fit(X)


inds,pts,_,_,_ = ens.query(np.array([0,0,0]),np.array([0.5,.1,.1]),idxs[0])
print(len(inds))
print(len(pts))
#print(inds)

inds,pts,_,_,_ = ens.query(np.array([0,0,0,0]),np.array([0.5,0.1,0.1,0.1]),idxs[1])

print(len(inds))
print(len(pts))

inds,pts,_,_,_ = ens.query(np.array([0,0,0]),np.array([0.5,.1,.1]),idxs[0])
inds,pts,_,_,_ = ens.query(np.array([0.5,0.5,0.3]),np.array([0.5,0.5,0.4]),idxs[2])

inds,cnts,_,_,_ = ens.multi_query_ranked(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.3],[0.5,0.5,0.4]]),[idxs[i] for i in [0,2]])
print(len(inds))
print(len(np.unique(inds)))
print(cnts)

inds,pts,_,_,_ = ens.multi_query(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.3],[0.5,0.5,0.4]]),[idxs[i] for i in [0,2]],n_jobs=1)
print(len(inds))
print(len(np.unique(inds)))
print(pts)


#ens.compress_models()