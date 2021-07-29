import numpy as np 
from py_kdtree.kdtree import KDTree
from py_kdtree.treeset import KDTreeSet
import time
np.random.seed(42)

X = np.random.randn(100000,7).astype(np.float32)

idxs = [[0,1,2],[3,4,5,6],[0,1,3]]


ens = KDTreeSet(idxs,leaf_size=20,path="/home/cluelf/py_kdtree/run",dtype="float32",verbose=False)

ens.fit(X)

start = time.time()
inds,pts = ens.multi_query(np.array([[0,0,0],[0,0,0]]),np.array([[0.5,0.5,0.1],[0.1,0.1,0.2]]),[idxs[i] for i in [0,2]],n_jobs=1)
end = time.time()
print(len(inds))
print(end-start)