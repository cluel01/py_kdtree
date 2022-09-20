import numpy as np 
from py_kdtree.kdtree import KDTree
import time
np.random.seed(42)

X = np.random.randint(0,100,(100,2)).astype(np.float64)
#X = np.random.randint(0,100,(100000,3)).astype(np.float64)
print(len(np.unique(X,axis=0)))

tree = KDTree(leaf_size=10,path="./run",dtype="float64",inmemory=False)

tree.fit(X)

#inds,pts,lv,t = tree.query_box(np.array([0,0,0]),np.array([10,10,10]),index_only=False)
inds,distances,dtime = tree.query_point_cy(np.array([1,1],dtype=np.float64),k=5)
print(inds)
print(distances)
print(dtime)

print(X[inds])