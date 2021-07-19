import numpy as np 
from kdtree import KDTree

X = np.array([[1,1],[2,2],[3,4],[4,5],[1,-1],[1.5,1.5]])

tree = KDTree(X)

tree.query_box(np.array([0,0]),np.array([4,4]))