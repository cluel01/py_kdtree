import pstats, cProfile
import numpy as np 
from py_kdtree.kdtree import KDTree
import time
np.random.seed(42)

X = np.random.randint(0,100,(100000000,3)).astype(np.float64)
#X = np.random.randint(0,100,(100000,3)).astype(np.float64)


tree = KDTree(leaf_size=2000,path="/home/cluelf/py_kdtree/run",dtype="float64")

tree.fit(X)

def test_func():
    #inds,pts,lv,t = tree.query_box(np.array([0,0,0]),np.array([10,10,10]),index_only=False)
    inds,t = tree.query_box_cy(np.array([0,0,0],dtype=np.float64),np.array([10,10,10],dtype="float64"),mem_cap=0.0001)

cProfile.runctx("test_func()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()