import time
import numpy as np
import math
import os
import pickle

# we generate random numbers; setting a "seed"
# will lead to the same "random" set when 
# exexuting the cell mulitple times
np.random.seed(42)

class Node():
    """ Helper class that reprents
    a single node of a k-d tree.
    """
    
    def __init__(self, left, right, points=None,bounds=None,dtype=None,path=None):
        self.left = left
        self.right = right
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
    def __init__(self, path=None,dtype="float64",leaf_size=30,model_file=None):
        if path is None:
            path = os.getcwd()
        
        tmp_path = path +"/.mmap"
        
        if not os.path.isdir(path):
            os.makedirs(path)

        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)

        self.path = path
        self.tmp_path = tmp_path

        self.dtype = dtype
        self.leaf_size = leaf_size

        self._root = None
        
        if model_file is None:
            self.model_file = os.path.join(path,"tree.pkl")
            if os.path.isfile(self.model_file):
                print(f"INFO: Load existing model under {self.model_file}")
                self._load()
        else:
            self.model_file = os.path.join(path,model_file)
            print(f"INFO: Load existing model under {self.model_file}")
            self._load()

        assert self.leaf_size == leaf_size, "Leaf size of model needs to match the input!"
    
    def fit(self, X):
        self._dim = len(X[0])
        
        assert np.dtype(self.dtype) == X.dtype, f"X dtype {X.np.dtype} does not match with Model dtype {self.np.dtype}"

        if len(os.listdir(self.tmp_path)) > 0:
            filelist = [ f for f in os.listdir(self.tmp_path) if f.endswith(".mmap") ]
            for f in filelist:
                os.remove(os.path.join(self.tmp_path, f))

        I = np.array(range(len(X)))
        #points = X.copy()
        start = time.time()
        self._root = self._build_tree(X, I)
        end = time.time()
        self._save()
        print(f"INFO: Building tree took {end-start} seconds")


    def _build_tree(self, pts, indices, depth=0,bounds=None):
        if bounds is None:
            bounds = np.array([[-np.inf,np.inf]]*self._dim)

        if len(pts) <= self.leaf_size: 
            pts = np.c_[indices,pts]
            return Node(left=None, right=None, points=pts,bounds=bounds,dtype=self.dtype,
                        path=self.tmp_path)
        
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

    def _recursive_search(self,node,mins,maxs,indices=None,points=None):
        if points is None:
            #points = np.empty((0,self._dim))
            points = []
        if indices is None:
            indices = []
        if (node.left == None and node.right==None):
            # is partition fully contained by box
            if (np.all(node.bounds[:,0] >= mins)) and (np.all(node.bounds[:,1] <= maxs)):
                pts = node._get_pts()
                indices.extend(list(pts[:,0].astype(np.int64)))
                points.extend(pts[:,1:])
                return indices,points
            #intersects
            elif not ( np.any(node.bounds[:,0] > maxs) ) or ( np.any(node.bounds[:,1] < mins )):
                pts = node._get_pts()
                mask = (np.all(pts[:,1:] >= mins,axis=1) ) &  (np.all(pts[:,1:] <= maxs, axis=1))
                indices.extend(list(pts[:,0][mask].astype(np.int64)))
                points.extend(pts[:,1:][mask])
                return indices,points
            else:
                return indices,points

        l_bounds = node.left.bounds 
        r_bounds = node.right.bounds

        #if at least intersects
        if not ( np.any(l_bounds[:,0] > maxs) ) or ( np.any(l_bounds[:,1] < mins )):
            self._recursive_search(node.left,mins,maxs,indices,points)

        if not ( np.any(r_bounds[:,0] > maxs) ) or ( np.any(r_bounds[:,1] < mins )):
            self._recursive_search(node.right,mins,maxs,indices,points)

        return indices,points

    def _load(self):
        with open(self.model_file, 'rb') as file:
            new = pickle.load(file)
            #self = pickle.load(file)
        self.__dict__.update(new.__dict__)

    def _save(self):
        with open(self.model_file, 'wb') as file:
            pickle.dump(self, file) 
        print(f"Model was saved under {self.model_file}")




        


            
        


