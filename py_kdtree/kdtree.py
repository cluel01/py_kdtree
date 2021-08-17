import time
import numpy as np
import math
import os
import pickle

# we generate random numbers; setting a "seed"
# will lead to the same "random" set when 
# exexuting the cell mulitple times
np.random.seed(42)
          

class KDTree():
    def __init__(self, path=None,dtype="float64",leaf_size=30,model_file=None,mmap_file=None,verbose=True):
        if path is None:
            path = os.getcwd()
        
        if not os.path.isdir(path):
            os.makedirs(path)

        self.path = path
        self.verbose = verbose
        self.dtype = dtype
        self.leaf_size = leaf_size

        self.tree = None

        if mmap_file is None:
            self.mmap_file = os.path.join(self.path,"map.mmap")
        else:
            self.mmap_file = os.path.join(self.path,mmap_file)
        
        if model_file is None:
            self.model_file = os.path.join(path,"tree.pkl")
        else:
            self.model_file = os.path.join(path,model_file)

        if os.path.isfile(self.model_file) and os.path.isfile(self.mmap_file):
            if self.verbose:
                print(f"INFO: Load existing model under {self.model_file}")
            self._load()

            #keep track if the input leaf size matches the loaded 
            self.org_leaf_size = leaf_size

    
    def fit(self, X):
        self._dim = len(X[0])
        
        assert np.dtype(self.dtype) == X.dtype, f"X dtype {X.dtype} does not match with Model dtype {self.dtype}"

        if self.tree is not None:
            if self.verbose:
                print("INFO: Model is already loaded, overwrite existing model!")
            os.remove(self.model_file)
            os.remove(self.mmap_file)

            self.leaf_size = self.org_leaf_size

        I = np.array(range(len(X)))

        self.depth = self._calc_depth(len(X))
        #update the leaf size with the actual value
        self.leaf_size = int(np.ceil(len(X) / 2**self.depth))
        self.n_leaves = 2**self.depth
        self.n_nodes = 2**(self.depth+1)-1
        self.tree = np.empty((self.n_nodes,self._dim,2),dtype=self.dtype)
        self.mmap_shape = (self.n_leaves,self.leaf_size,self._dim+1)

        mmap = np.memmap(self.mmap_file, dtype=self.dtype, mode='w+', shape=self.mmap_shape)

        start = time.time()
        self._build_tree(X, I,mmap)
        end = time.time()
        #mmap.flush()
        self._save()
        if self.verbose:
            print(f"INFO: Building tree took {end-start} seconds")


    def _build_tree(self, pts, indices,mmap, depth=0,idx=0):
        #if root
        if idx == 0: 
            self.tree[idx] = np.array([[-np.inf,np.inf]]*self._dim)

        bounds = self.tree[idx]

        if len(pts) <= self.leaf_size: 
            pts = np.c_[indices,pts]

            shape = pts.shape
            if shape[0] != self.leaf_size:
                nan = np.array([-1,*[-np.inf]*self._dim],dtype=self.dtype)
                pts = np.vstack([pts,nan])
            
            lf_idx = self.n_leaves+idx-self.n_nodes
            mmap[lf_idx,:] = pts[:]
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

        self._build_tree(pts[:midx,:], indices[:midx],mmap, depth+1,l_idx)
        self._build_tree(pts[midx:,:], indices[midx:],mmap, depth+1,r_idx)


    def query_box(self,mins,maxs):
        if self.tree is None:  
            raise Exception("Tree not fitted yet!")

        start = time.time()
        indices,points = self._recursive_search(0,mins,maxs)
        end = time.time()
        if self.verbose:
            print(f"INFO: Box search took: {end-start} seconds")
        return indices,np.array(points,dtype=self.dtype)

    def _recursive_search(self,idx,mins,maxs,indices=None,points=None):
        if points is None:
    
            points = []
        if indices is None:
            indices = []

        l_idx,r_idx = self._get_child_idx(idx)
        
        if (l_idx >= len(self.tree)) and (r_idx >= len(self.tree)):
            bounds = self.tree[idx]
            #intersects
            if (np.all(bounds[:,1] >= mins )) and (np.all(maxs >= bounds[:,0])):
                lf_idx = self.n_leaves+idx-self.n_nodes
                pts = self._get_pts(lf_idx)
                mask = (np.all(pts[:,1:] >= mins,axis=1) ) &  (np.all(pts[:,1:] <= maxs, axis=1))
                indices.extend(pts[:,0][mask].astype(np.int64))
                points.extend(pts[:,1:][mask])
                return indices,points
            else:
                return indices,points

        l_bounds = self.tree[l_idx]
        r_bounds = self.tree[r_idx]

        #if at least intersects
        if (np.all(l_bounds[:,1] >= mins )) and (np.all(maxs >= l_bounds[:,0])):
            self._recursive_search(l_idx,mins,maxs,indices,points)

        if (np.all(r_bounds[:,1] >= mins )) and (np.all(maxs >= r_bounds[:,0])):
            self._recursive_search(r_idx,mins,maxs,indices,points)

        return indices,points

    def _load(self):
        with open(self.model_file, 'rb') as file:
            new = pickle.load(file)

        self.tree = new.tree

        #TODO merge attributes into one
        self.depth = new.depth
        self.leaf_size = new.leaf_size
        self.n_leaves = new.n_leaves
        self.n_nodes = new.n_nodes
        self._dim = new._dim
        self.mmap_shape = new.mmap_shape
        self.dtype = str(self.tree.dtype)

    def _save(self):
        with open(self.model_file, 'wb') as file:
            pickle.dump(self, file) 
        if self.verbose:            
            print(f"Model was saved under {self.model_file}")

    def _calc_depth(self,n):
        d = 0
        while n/2**d > self.leaf_size:
            d += 1
        return d

    def _get_pts(self,lf_idx):
        fp = np.memmap(self.mmap_file, dtype=self.dtype, mode='r', shape=self.mmap_shape) 
        return fp[lf_idx,:,:]

    #Returns config parameters of model required to load the model
    def get_file_cfg(self):
        return {"path":self.path,"mmap_file":self.mmap_file,"model_file":self.model_file,
                "verbose":self.verbose} 
            
    @staticmethod
    def _get_child_idx(i):
        return (2*i)+1, (2*i)+2






        


            
        


