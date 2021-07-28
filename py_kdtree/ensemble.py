from operator import index
import numpy as np
import os
import h5py

from .kdtree import KDTree

class KDTreeEnsemble():
    def __init__(self,indexes,model_file=None,path=None,**kwargs) -> None:       
        if isinstance(indexes,np.ndarray):
            if len(indexes.shape) == 2:
                indexes = indexes.tolist()
            else:
                raise Exception("Indexes needs to be a 2D array!")
        elif not isinstance(indexes,list):
            raise Exception("No known datatype for indexes")
        
        self.indexes = indexes
        self.n = len(indexes)

        self.trees = {}

        self.path = path
        if path is None:
            self.path = os.getcwd()

        if model_file is None:
            self.model_file = os.path.join(path,"kdtree_ensemble.h5")
        else:
            if os.path.isabs(model_file):
                self.model_file = model_file
            else:
                self.model_file = os.path.join(path,model_file)

        if not os.path.isfile(self.model_file):
            self.h5f = h5py.File(self.model_file, 'w')
            #Not trained flag
            self.h5f.attrs["trained"] = 0

            for i in indexes:
                gname = "_".join([str(j) for j in i])
                grp = self.h5f.create_group(gname)
                grp.attrs["features"] = i

                t = KDTree(path=self.path,model_file=os.path.basename(self.model_file),h5group=gname,**kwargs)
                self.trees[gname] = t
        else:
            self.h5f = h5py.File(self.model_file, 'a')
            for i in indexes:
                gname = "_".join([str(j) for j in i])
                if not gname in self.h5f:
                    raise Exception(f"Index {str(i)} is missing in existing file {self.model_file}. Create new model with the required indexes by providing a different <model_file>!")
                self.trees[gname] = KDTree(self.path,model_file=os.path.basename(self.model_file),h5group=gname)

    def __len__(self):
        return self.n

    def fit(self,X):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

        for i in self.indexes:
            gname = "_".join([str(j) for j in i])
            self.trees[gname].fit(X[:,i])
        self.h5f.attrs["trained"] = 1



    # Fitting the trees in sequential manner when X is too large to fit into memory as a whole
    def fit_seq(self,X_parts_list):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

    def query(self,mins,maxs,idx):
        assert self.h5f.attrs["trained"] == 1,"Group of trees needs to be trained first!"

        #query stuff

    def multi_query(self,mins,maxs,idxs):
        pass