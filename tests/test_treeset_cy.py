import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.treeset import KDTreeSet
np.random.seed(42)

X = np.random.randn(100000,7).astype(np.float64)

idxs = [[0,1,2],[3,4,5,6],[0,1,3],[1,3,0]]


ens = KDTreeSet(idxs,leaf_size=40,path="/home/cluelf/py_kdtree/run",dtype="float64",group_prefix="",verbose=True)

ens.fit(X)


inds,cnts,_,_,_ = ens.multi_query_ranked(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.3],[0.5,0.5,0.4]]),[idxs[i] for i in [0,2]])
print(len(inds))
print(len(np.unique(inds)))

inds,cnts,_ = ens.multi_query_ranked_cy(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.3],[0.5,0.5,0.4]]),[idxs[i] for i in [2,3]])
print(len(inds))
print(len(np.unique(inds)))

# inds,cnts,_ = ens.multi_query_ranked_parallel_cy(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.3],[0.5,0.5,0.4]]),[idxs[i] for i in [0,2]],n_jobs=4)
# print(len(inds))
# print(len(np.unique(inds)))

