import numpy as np
import os
import h5py
import time
import multiprocessing
import os 
import sys
from .kdtree import KDTree

class KDTreeSet():
    def __init__(self,indexes,model_file=None,path=None,dtype="float64",verbose=True,**kwargs) -> None:       
        if isinstance(indexes,np.ndarray):
            if len(indexes.shape) == 2:
                indexes = indexes.tolist()
            else:
                raise Exception("Indexes needs to be a 2D array!")
        elif not isinstance(indexes,list):
            raise Exception("No known datatype for indexes")
        
        self.indexes = indexes
        self.n = len(indexes)
        self.verbose = verbose
        self.trees = {}

        self.path = path
        if path is None:
            self.path = os.getcwd()

        if model_file is None:
            self.model_file = os.path.join(path,"kdtree_set.h5")
        else:
            if os.path.isabs(model_file):
                self.model_file = model_file
            else:
                self.model_file = os.path.join(path,model_file)

        if not os.path.isfile(self.model_file):
            self.h5f = h5py.File(self.model_file, 'w')
            #Not trained flag
            self.h5f.attrs["trained"] = 0
            self.h5f.attrs["dtype"] = dtype
            

            for i in indexes:
                gname = "_".join([str(j) for j in i])
                grp = self.h5f.create_group(gname)
                grp.attrs["features"] = i

                t = KDTree(path=self.path,model_file=os.path.basename(self.model_file),h5group=gname,dtype=dtype,verbose=verbose,
                            **kwargs)
                self.trees[gname] = t
        else:
            self.h5f = h5py.File(self.model_file, 'a')
            dtype = self.h5f.attrs["dtype"]
            for i in indexes:
                gname = "_".join([str(j) for j in i])
                if not gname in self.h5f:
                    raise Exception(f"Index {str(i)} is missing in existing file {self.model_file}. Create new model with the required indexes by providing a different <model_file>!")
                self.trees[gname] = KDTree(self.path,model_file=os.path.basename(self.model_file),h5group=gname,dtype=dtype,verbose=verbose)
        self.dtype = dtype

    def __len__(self):
        return self.n

    def fit(self,X):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"
        if self.h5f.attrs["trained"] == 0:
            for i in self.indexes:
                gname = "_".join([str(j) for j in i])
                self.trees[gname].fit(X[:,i])
            self.h5f.attrs["trained"] = 1
        else:
            if self.verbose:
                print("INFO: Skipping train as the model has already been trained! Change model_file in case of a new model!")



    # Fitting the trees in sequential manner when X is too large to fit into memory as a whole
    def fit_seq(self,X_parts_list,parts_path=None):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

        if self.h5f.attrs["trained"] == 0:
            if parts_path is None:
                parts_path = self.path

            for i in self.indexes:
                data = []
                for f in X_parts_list:
                    fname = os.path.join(parts_path,f)
                    x = np.load(fname)[:,i]
                    if x.dtype != np.dtype(self.dtype):
                        x = x.astype(self.dtype)
                    data.append(x)

                X = np.vstack(data)          
                gname = "_".join([str(j) for j in i])
                self.trees[gname].fit(X)

            self.h5f.attrs["trained"] = 1
        else:
            if self.verbose:
                print("INFO: Skipping train as the model has already been trained! Change model_file in case of a new model!")

    def query(self,mins,maxs,idx):
        assert self.h5f.attrs["trained"] == 1,"Group of trees needs to be trained first!"

        if mins.dtype != np.dtype(self.dtype):
            mins = mins.astype(self.dtype)
        if maxs.dtype != np.dtype(self.dtype):
            maxs = maxs.astype(self.dtype)

        #query stuff
        gname = "_".join([str(j) for j in idx])
        inds, pts = self.trees[gname].query_box(mins,maxs)

        return inds,pts

    '''
    Input:
    mins and maxs : arrays or lists of min/max boundaries (in 2D array format) 
                    -> lists required for varying dims of indices
    '''
    
    #TODO parallelization? -> problematic due to h5py objects in class
    def multi_query(self,mins,maxs,idxs,no_pts=False):
        assert self.h5f.attrs["trained"] == 1,"Group of trees needs to be trained first!"

        if isinstance(mins,np.ndarray):
            if mins.dtype != np.dtype(self.dtype):
                mins = mins.astype(self.dtype)

        if isinstance(maxs,np.ndarray):       
            if maxs.dtype != np.dtype(self.dtype):
                maxs = maxs.astype(self.dtype)

        start = time.time()
        i_list = []
        # To account for returned pts of different dimensionality 
        p_list = []
        for i in range(len(idxs)):
            inds, pts =self.query(mins[i],maxs[i],idxs[i])
            #get inds not part of i_list so far
            new_idx = np.where(np.in1d(inds,i_list) == False)
            i_list.extend(np.array(inds)[new_idx])
            if no_pts == False:
                p_list.append(pts[new_idx])
        end = time.time()

        if self.verbose:
            print(f"INFO: query finished in {end-start} seconds")

        return i_list,p_list
            





