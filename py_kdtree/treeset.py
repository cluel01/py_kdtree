import numpy as np
import os
import time
import sys
import os 
import multiprocessing
from .kdtree import KDTree

class KDTreeSet():
    def __init__(self,indexes,path=None,dtype="float64",model_name="tree.pkl",verbose=True,**kwargs) -> None:       
        if isinstance(indexes,np.ndarray):
            if len(indexes.shape) == 2:
                indexes = indexes.tolist()
            else:
                raise Exception("Indexes needs to be a 2D array!")
        elif not isinstance(indexes,list):
            raise Exception("No known datatype for indexes")
        
        self.indexes = indexes
        self.n = len(indexes)
        self.model_name = model_name
        self.verbose = verbose
        self.dtype = dtype

        self.trees = {}

        self.path = path
        if path is None:
            self.path = os.getcwd()


        # Check if all models exist
        exists = True
        for i in indexes:
            dname = "_".join([str(j) for j in i])
            full = os.path.join(path,dname)

            if not os.path.isdir(full):
                exists = False
                break
            else:
                tree_file = os.path.join(full,model_name)
                if os.path.isfile(tree_file):
                    exists = False
                    break
                
        self.trees = {}
        if exists:
            for i in indexes:
                dname = "_".join([str(j) for j in i])
                full = os.path.join(path,dname)
                tree = KDTree(path=full,model_file=model_name,verbose=self.verbose)
                self.trees[dname] = tree

        else:
            for i in indexes:
                dname = "_".join([str(j) for j in i])
                full = os.path.join(path,dname)
                tree = KDTree(path=full,dtype=dtype,model_file=model_name,verbose=self.verbose,**kwargs)
                self.trees[dname] = tree        


    def __len__(self):
        return self.n

    def fit(self,X):    
        for i in self.indexes:
            dname = "_".join([str(j) for j in i])
            if self.trees[dname].tree is not None:
                if self.verbose:
                    print("INFO: Skip tree fit, model already existing - Change <path> in case of a new model!")
            else:
                self.trees[dname].fit(X[:,i])

    # Fitting the trees in sequential manner when X is too large to fit into memory as a whole
    def fit_seq(self,X_parts_list,parts_path=None):
        if parts_path is None:
            parts_path = self.path

        for i in self.indexes:
            dname = "_".join([str(j) for j in i])
            if self.trees[dname].tree is not None:
                if self.verbose:
                    print("INFO: Skip tree fit, model already existing - Change <path> in case of a new model!")
            else:
                data = []
                for f in X_parts_list:
                    fname = os.path.join(parts_path,f)
                    x = np.load(fname)[:,i]
                    if x.dtype != np.dtype(self.dtype):
                        x = x.astype(self.dtype)
                    data.append(x)

                X = np.vstack(data)    
                self.trees[dname].fit(X)

    def query(self,mins,maxs,idx):
        if mins.dtype != np.dtype(self.dtype):
            mins = mins.astype(self.dtype)
        if maxs.dtype != np.dtype(self.dtype):
            maxs = maxs.astype(self.dtype)

        #query stuff
        dname = "_".join([str(j) for j in idx])
        inds, pts = self.trees[dname].query_box(mins,maxs)

        return inds,pts

    '''
    Input:
    mins and maxs : arrays or lists of min/max boundaries (in 2D array format) 
                    -> lists required for varying dims of indices
    '''
    
    def multi_query(self,mins,maxs,idxs,no_pts=False,n_jobs=-1):
        if isinstance(mins,np.ndarray):
            if mins.dtype != np.dtype(self.dtype):
                mins = mins.astype(self.dtype)

        if isinstance(maxs,np.ndarray):       
            if maxs.dtype != np.dtype(self.dtype):
                maxs = maxs.astype(self.dtype)

        start = time.time()

        if n_jobs == -1:
            total_cpus = os.cpu_count()
            if total_cpus > len(idxs):
                n_jobs = len(idxs)
            else:
                n_jobs = total_cpus
        else:
            n_jobs = n_jobs

        params = [x for i in range(len(idxs))  for x in [[mins[i],maxs[i],idxs[i]]]]

        pool = multiprocessing.Pool(n_jobs)

        try:
            results = pool.starmap(self.query,params)
        except  Exception as e:
            print(f"Warning: Error in query! \n {e}")
            pool.close()
            sys.exit()
        pool.close()
        pool.join()

        i_list = []
        # To account for returned pts of different dimensionality 
        p_list = []
        for i in range(len(idxs)):
            inds, pts = results[i]
            #get inds not part of i_list so far
            new_idx = np.where(np.in1d(inds,i_list) == False)
            i_list.extend(np.array(inds)[new_idx])
            if no_pts == False:
                p_list.append(pts[new_idx])
        end = time.time()
        if self.verbose:
            print(f"INFO: query finished in {end-start} seconds")

        return i_list,p_list
            





