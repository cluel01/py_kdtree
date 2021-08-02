import numpy as np 
from py_kdtree.kdtree import KDTree
np.random.seed(42)

X = np.random.randn(1000000,3).astype(np.float32)

tree = KDTree(leaf_size=20000,path="/home/cluelf/py_kdtree/run",dtype="float32")

tree.fit(X)


print(tree.tree[-1][:,0])
mins = tree.tree[-1][:,0]+0.0001
print(mins)

inds,pts = tree.query_box(mins,tree.tree[-1][:,1])

print(tree.tree[-1])
print(len(inds))
#tree.compress_model()