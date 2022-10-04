from lib2to3.pytree import Leaf
import numpy as np 
from py_kdtree.kdtree import KDTree
from sklearn.neighbors import KDTree as KD
import time
np.random.seed(42)

leaf_size = 5

dtype = "float32"

X = np.random.randint(0,100,(100,2)).astype(dtype)
#X = np.random.randint(0,100,(100000,3)).astype(np.float64)
print(len(np.unique(X,axis=0)))

tree = KDTree(leaf_size=leaf_size,path="./run",dtype=dtype,inmemory=False)

tree.fit(X)

point = np.array([20,5],dtype=dtype)

print(tree.leaf_size)
#inds,pts,lv,t = tree.query_box(np.array([0,0,0]),np.array([10,10,10]),index_only=False)
inds,distances,dtime,leaves_visited = tree.query_point_cy(point,k=5,stop_leaves=10)
print(inds)
print(distances)
print(dtime)

print(X[inds])

t = KD(X,leaf_size)
start = time.time()
dist, ind = t.query(np.expand_dims(point, axis=0), k=5)  
end = time.time()
print(ind)
print(dist)
print(end-start)


# #inds,pts,lv,t = tree.query_box(np.array([0,0,0]),np.array([10,10,10]),index_only=False)
# inds,distances,dtime = tree.query_point(point,k=5)
# print(inds)
# print(distances)
# print(dtime)

# print(X[inds])