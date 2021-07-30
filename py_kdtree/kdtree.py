import time
import numpy as np
import math
import os

import h5py

# we generate random numbers; setting a "seed"
# will lead to the same "random" set when 
# exexuting the cell mulitple times
np.random.seed(42)
          

class KDTree():
    def __init__(self, path=None,dtype="float64",leaf_size=30,model_file=None,chunksize=None,compression=None,shuffle=None,h5group=None,verbose=True):
        if path is None:
            path = os.getcwd()
               
        if not os.path.isdir(path):
            os.makedirs(path)


        self.path = path
        self.verbose = verbose
        self.dtype = dtype
        self.leaf_size = leaf_size

        if chunksize is None:
            #autochunking enabled
            self.chunksize = True
        elif chunksize == "leaf":
            self.chunksize = None
        else:
            self.chunksize = chunksize

        self.compression = compression
        self.shuffle = shuffle

        self.h5group = h5group

        self.tree = None
        self.trained = False

        if model_file is None:
            self.model_file = os.path.join(path,"kdtree.h5")
        else:
            self.model_file = os.path.join(path,model_file)

        if os.path.isfile(self.model_file):
            h5f = h5py.File(self.model_file, 'a')
            keys = None
            if h5group is None:
                dt = h5f
            else:
                if "/" + h5group in h5f:
                    dt = h5f[h5group]
                else:
                    dt = h5f.create_group(h5group)
                    
            keys = list(dt.keys())
            if ("tree" in keys) and ("leaves" in keys):
                if self.verbose:
                    print(f"INFO: Load existing model!")
                self.tree = dt["tree"][()]
                self.n_nodes = dt.attrs["n_nodes"]
                self.n_leaves = dt.attrs["n_leaves"]
                self.leaf_size = dt.attrs["leaf_size"]
                self.trained = True

        else:
            if h5group is not None:
                raise Exception("WARNING: h5group only relevant for ensembles!")
            h5f,dt = self._init_h5("w")
        h5f.close()
        
    
    def fit(self, X):
        self._dim = len(X[0])
        
        assert np.dtype(self.dtype) == X.dtype, f"X dtype {X.dtype} does not match with Model dtype {self.dtype}"

        if self.tree is not None:
            #Only remove if it is a single model
            if self.h5group is None:
                if self.verbose:
                    print("INFO: Model is already loaded, overwrite existing model!")
                os.remove(self.model_file)
            else:
                raise Exception("Model already fitted!")

        h5f,dt = self._init_h5("a")

        I = np.array(range(len(X)))

        self.depth = self._calc_depth(len(X))
        #updated with the actual leaf size
        self.leaf_size = int(np.ceil(len(X) / 2**self.depth))
        self.n_leaves = 2**self.depth
        self.n_nodes = 2**(self.depth+1)-1

        dt.attrs["n_nodes"] = self.n_nodes
        dt.attrs["n_leaves"] = self.n_leaves
        dt.attrs["leaf_size"] = self.leaf_size

        if self.chunksize is None:
            self.chunksize = (1,self.leaf_size,self._dim+1)

        self.tree = dt.create_dataset("tree",shape=(self.n_nodes,self._dim,2),dtype=self.dtype,compression=self.compression,shuffle=self.shuffle)

        self.leaves = dt.create_dataset("leaves",shape=(self.n_leaves,self.leaf_size,self._dim+1),dtype=self.dtype,chunks=self.chunksize,
                                                compression=self.compression,shuffle=self.shuffle)
        
        start = time.time()
        self._build_tree(X, I)
        end = time.time()
        self.tree = self.tree[()]

        self.trained = True

        if self.verbose:
            print(f"INFO: Building tree took {end-start} seconds")
        h5f.close()


    def _build_tree(self, pts, indices, depth=0,idx=0):
        #if root
        if idx == 0: 
            self.tree[idx] = np.array([[-np.inf,np.inf]]*self._dim)

        bounds = self.tree[idx]

        if len(pts) <= self.leaf_size: 
            pts = np.c_[indices,pts]

            shape = pts.shape
            if shape[0] != self.leaf_size:
                nan = np.array([-1,*[-np.inf]*self._dim],self.dtype)
                pts = np.vstack([pts,nan])
            lf_idx = self.n_nodes-self.n_leaves-idx
            self.leaves[lf_idx] = pts
            return 
        
        axis = depth % self._dim
        
        part = pts[:,axis].argsort()
        indices = indices[part]
        pts = pts[part]

        midx = math.floor(len(pts)/2)
        median = pts[midx, axis]

        l_bounds,r_bounds = bounds.copy(),bounds.copy()
        l_bounds[axis,1] = median
        r_bounds[axis,0] = median

        l_idx,r_idx = self._get_child_idx(idx)

        self.tree[l_idx] = l_bounds
        self.tree[r_idx] = r_bounds

        self._build_tree(pts[:midx,:], indices[:midx], depth+1,l_idx)
        self._build_tree(pts[midx:,:], indices[midx:], depth+1,r_idx)


    def query_box(self,mins,maxs):
        if self.tree is None:  
            raise Exception("Tree not fitted yet!")

        start = time.time()

        h5f, dt = self._init_h5("r")

        self.leaves = dt["leaves"]

        indices,points = self._recursive_search(0,mins,maxs)
        end = time.time()
        if self.verbose:
            print(f"INFO: Box search took: {end-start} seconds")

        h5f.close()
        
        return indices,np.array(points)

    def _recursive_search(self,idx,mins,maxs,indices=None,points=None):
        if points is None:
            points = []
        if indices is None:
            indices = []

        l_idx,r_idx = self._get_child_idx(idx)
        
        #if leaf
        if (l_idx >= len(self.tree)) and (r_idx >= len(self.tree)):
            # is partition fully contained by box
            bounds = self.tree[idx]
            if (np.all(bounds[:,0] >= mins)) and (np.all(bounds[:,1] <= maxs)):
                lf_idx = self.n_nodes-self.n_leaves-idx
                pts = self.leaves[lf_idx]
                indices.extend(pts[:,0].astype(np.int64))
                points.extend(pts[:,1:])
                return indices,points
            #intersects
            elif not ( np.any(bounds[:,0] > maxs) ) or ( np.any(bounds[:,1] < mins )):
                lf_idx = self.n_nodes-self.n_leaves-idx
                pts = self.leaves[lf_idx]
                mask = (np.all(pts[:,1:] >= mins,axis=1) ) &  (np.all(pts[:,1:] <= maxs, axis=1))
                indices.extend(pts[:,0][mask].astype(np.int64))
                points.extend(pts[:,1:][mask])
                return indices,points
            else:
                return indices,points

        l_bounds = self.tree[l_idx]
        r_bounds = self.tree[r_idx]

        #if at least intersects
        if not ( np.any(l_bounds[:,0] > maxs) ) or ( np.any(l_bounds[:,1] < mins )):
            self._recursive_search(l_idx,mins,maxs,indices,points)

        if not ( np.any(r_bounds[:,0] > maxs) ) or ( np.any(r_bounds[:,1] < mins )):
            self._recursive_search(r_idx,mins,maxs,indices,points)

        return indices,points

    def _calc_depth(self,n):
        d = 0
        while n/2**d > self.leaf_size:
            d += 1
        return d

    def _init_h5(self,mode):
        h5f = h5py.File(self.model_file, mode)
        if self.h5group is None:
            dt = h5f
        else:
            dt = h5f[self.h5group]
        return h5f,dt

    @staticmethod
    def _get_child_idx(i):
        return (2*i)+1, (2*i)+2 




        


            
        


