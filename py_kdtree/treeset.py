import numpy as np
import os
import h5py
import time
import multiprocessing
import os 
import sys
from .kdtree import KDTree

class KDTreeSet():
    def __init__(self,indexes,model_file=None,path=None,dtype="float64",verbose=True,group_prefix="",**kwargs) -> None:       
        if isinstance(indexes,np.ndarray):
            if len(indexes.shape) == 2:
                indexes = indexes.tolist()
            else:
                raise Exception("Indexes needs to be a 2D array!")
        elif not isinstance(indexes,list):
            raise Exception("No known datatype for indexes")

        #remove duplicates
        idx_set = set(tuple(x) for x in indexes)
        self.indexes = [ list(x) for x in idx_set ]

        self.n = len(indexes)
        self.verbose = verbose
        self.trees = {}

        self.dtype = dtype
        self.group_prefix = group_prefix

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
            h5f = h5py.File(self.model_file, 'w')
            h5f.attrs["dtype"] = dtype
            h5f.attrs["group_prefix"] = group_prefix
        else:
            h5f = h5py.File(self.model_file, 'a')
            self.dtype = h5f.attrs["dtype"]
            self.group_prefix = h5f.attrs["group_prefix"]
            
        self.trained = True
        for i in indexes:
            gname = "_".join([self.group_prefix + str(j) for j in i])

            if not gname in h5f:
                grp = h5f.create_group(gname)
                grp.attrs["features"] = i
            
            t = KDTree(path=self.path,model_file=os.path.basename(self.model_file),h5group=gname,dtype=self.dtype,verbose=self.verbose,
                        **kwargs)
            if not t.trained:
                self.trained = False
            self.trees[gname] = t

        h5f.close()

    def __len__(self):
        return self.n

    def fit(self,X):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

        if not self.trained:
            for i in self.indexes:
                gname = "_".join([self.group_prefix + str(j) for j in i])
                tree = self.trees[gname]
                if not tree.trained:
                    tree.fit(X[:,i])
                else:
                    if self.verbose:
                        print(f"INFO: skipping model {gname} as it is already trained!")
            self.trained = True
        else:
            if self.verbose:
                print("INFO: Skipping train as the model has already been trained! Change model_file in case of a new model!")




    # Fitting the trees in sequential manner when X is too large to fit into memory as a whole
    '''
    ncached_idx - defines the number of indices are stored at once in memory
    '''

    def fit_seq(self,X_parts_list,parts_path=None,n_chached=1):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

        if not self.trained:
            if parts_path is None:
                parts_path = self.path

            #Filter trained idxs
            idxs = []
            for i in self.indexes:
                gname = "_".join([self.group_prefix + str(j) for j in i])
                if self.trees[gname].trained == False:
                    idxs.append(i)

            c = 0
            while c < len(idxs):
                sub = idxs[c:c+n_chached]
                flat_idx = [item for sublist in sub for item in sublist] #Flatten list
                data = []
                for f in X_parts_list:
                    fname = os.path.join(parts_path,f)
                    x = np.load(fname)[:,flat_idx]
                    if x.dtype != np.dtype(self.dtype):
                        x = x.astype(self.dtype)
                    data.append(x)
                X = np.vstack(data)

                #Train models
                for i in range(len(sub)):
                    start = len([item for sublist in sub[:i] for item in sublist])
                    end = start+len(sub[i])
                    gname = "_".join([self.group_prefix + str(j) for j in sub[i]])
                    self.trees[gname].fit(X[:,start:end])
                c += n_chached

            self.trained = True
        else:
            if self.verbose:
                print("INFO: Skipping train as the model has already been trained! Change model_file in case of a new model!") 

    def query(self,mins,maxs,idx):
        assert self.trained,"Group of trees needs to be trained first!"

        if mins.dtype != np.dtype(self.dtype):
            mins = mins.astype(self.dtype)
        if maxs.dtype != np.dtype(self.dtype):
            maxs = maxs.astype(self.dtype)

        #query stuff
        gname = "_".join([self.group_prefix + str(j) for j in idx])
        inds, pts = self.trees[gname].query_box(mins,maxs)

        return inds,pts

    '''
    Input:
    mins and maxs : arrays or lists of min/max boundaries (in 2D array format) 
                    -> lists required for varying dims of indices
    '''
    def multi_query(self,mins,maxs,idxs,no_pts=False,n_jobs=-1):
        assert self.trained,"Group of trees needs to be trained first!"

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

    
            





