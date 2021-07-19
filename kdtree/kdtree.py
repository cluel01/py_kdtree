import time
import numpy as np
import math
import os

# we generate random numbers; setting a "seed"
# will lead to the same "random" set when 
# exexuting the cell mulitple times
np.random.seed(42)

class Node():
    """ Helper class that reprents
    a single node of a k-d tree.
    """
    
    def __init__(self, left, right, points=None, indices=None,bounds=None,dtype=None,path=None):
        self.left = left
        self.right = right
        self.indices = indices
        self.bounds = bounds
        self.dtype = dtype
        self.path = path
        self.isLeaf = False

        filename = None
        #if leaf
        if points is not None:
            self.shape = points.shape
            filename = os.path.join(path, "mem"+str(time.time())+".mmap")
            fp = np.memmap(filename, dtype=dtype, mode='w+', shape=points.shape)
            fp[:] = points[:]
            fp.flush()

            self.isLeaf = True

        self.filename = filename

    def _get_pts(self,mask=None):
        assert self.isLeaf, "Node needs to be a leaf!"
        
        fp = np.memmap(self.filename, dtype=self.dtype, mode='r', shape=self.shape) 
        if mask is not None:
            fp = fp[mask]
        return fp
            

class KDTree():
    
    def __init__(self, X, path=None,dtype=None,leaf_size=30):
        if dtype is None:
            self.dtype = X.dtype 
        else:
            self.dtype = dtype

        if path is None:
            path = os.getcwd()+"/.mmap"
            if not os.path.isdir(path):
                os.makedirs(path)

        if len(os.listdir(path)) > 0:
            filelist = [ f for f in os.listdir(path) if f.endswith(".mmap") ]
            for f in filelist:
                os.remove(os.path.join(path, f))
        self.path = path

        self.leaf_size = leaf_size
        self._fit(X)    
    
    def _fit(self, X):
        
        self._dim = len(X[0])
        
        I = np.array(range(len(X)))
        #points = X.copy()
        self._root = self._build_tree(X, I)

    def _build_tree(self, pts, indices, depth=0,bounds=None):
        if bounds is None:
            bounds = np.array([[-np.inf,np.inf]]*self._dim)

        if len(pts) <= self.leaf_size: 
            return Node(left=None, right=None, points=pts, indices=indices,bounds=bounds,dtype=self.dtype,
                        path=self.path)
        
        axis = depth % self._dim
        
        part = pts[:,axis].argsort()
        indices = indices[part]
        pts = pts[part]

        midx = math.floor(len(pts)/2)
        median = pts[midx, axis]

        l_bounds,r_bounds = bounds.copy(),bounds.copy()
        l_bounds[axis,1] = median
        r_bounds[axis,0] = median

        lefttree = self._build_tree(pts[:midx,:], indices[:midx], depth+1,l_bounds)
        righttree = self._build_tree(pts[midx:,:], indices[midx:], depth+1,r_bounds)

        return Node(left=lefttree, right=righttree,bounds=bounds)

    def query_box(self,mins,maxs):
        if self._root is None:  
            raise Exception("Tree not fitted yet!")

        start = time.time()
        indices,points = self._recursive_search(self._root,mins,maxs)
        end = time.time()
        print(f"INFO: Box search took: {end-start} seconds")
        return indices,np.array(points)

    def _recursive_search(self,node,mins,maxs,indices=[],points=None):
        if points is None:
            #points = np.empty((0,self._dim))
            points = []
        if (node.left == None and node.right==None):
            # is partition fully contained by box
            if (np.all(node.bounds[:,0] >= mins)) and (np.all(node.bounds[:,1] <= maxs)):
                indices.extend(list(node.indices))
                points.extend(node._get_pts())
                return indices,points
            else:
                pts = node._get_pts()
                mask = (np.all(pts >= mins,axis=1) ) &  (np.all(pts <= maxs, axis=1))
                indices.extend(list(node.indices[mask]))
                points.extend(pts[mask])
                return indices,points

        l_bounds = node.left.bounds 
        r_bounds = node.right.bounds

        #if intersects
        if not ( np.any(l_bounds[:,0] > maxs) ) or ( np.any(l_bounds[:,1] < mins )):
            self._recursive_search(node.left,mins,maxs,indices,points)

        if not ( np.any(r_bounds[:,0] > maxs) ) or ( np.any(r_bounds[:,1] < mins )):
            self._recursive_search(node.right,mins,maxs,indices,points)

        return indices,points


        


        


            
        


