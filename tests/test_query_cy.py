import numpy as np 
from py_kdtree.kdtree import KDTree
import time
np.random.seed(42)

X = np.random.randint(0,100,(100000,3)).astype(np.float32)
#X = np.random.randint(0,100,(100000,3)).astype(np.float64)
print(len(np.unique(X,axis=0)))

tree = KDTree(leaf_size=10,path="./run",dtype="float32")

tree.fit(X)


inds,pts,lv,t,_ = tree.query_box(np.array([0,0,0]),np.array([10,10,10]),index_only=False)

#inds,t,_ = tree.query_box_cy(np.array([1,1,1],[0,0,0]],dtype=np.float32),np.array([[2,2,2],[10,10,10]],dtype="float32"),max_pts=600,max_leaves=0,mem_cap=0.0001)
#inds,_,_,_,_ = tree.query_box(np.array([0,0,0],dtype=np.float64),np.array([10,10,10],dtype="float64"))
#inds,_,_,_,_,_,_ = tree.query_box_cy_profile(np.array([0,0,0],dtype=np.float64),np.array([10,10,10],dtype="float64"),max_pts=100,mem_cap=0.0001)


print(len(inds))
print(len(np.unique(inds)))
print(inds[:10])
print(inds[-10:])
print(np.where(inds == -1))
u, c = np.unique(inds, return_counts=True)
print(u[c > 1])
#print(inds)
print("++++++++++++++++++++")
start = time.time()
res =  (np.all(X >= np.array([0,0,0]),axis=1) ) &  (np.all(X <= np.array([10,10,10]), axis=1))
end = time.time()
print(np.sum(res))
print(end-start)
print(len(np.unique(res[0])))

ret = (np.all(X>=np.array([0,0,0]),axis=1) & np.all(X <= np.array([10,10,10]), axis=1))
print(np.sum(ret))
inds_test = np.where(ret == True)[0]
print("Diff:",np.setdiff1d(inds,inds_test))