import numpy as np
#import torch
import os
import time
import sys
import os 
import multiprocessing
from .kdtree import KDTree

class KDTreeSet():
    def __init__(self,indexes,path=None,dtype="float64",model_name="tree.pkl",verbose=True,group_prefix="",**kwargs) -> None:       
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
        self.model_name = model_name
        self.verbose = verbose
        self.dtype = dtype
        self.group_prefix = group_prefix

        self.trees = {}

        self.path = path
        if path is None:
            self.path = os.getcwd()
                
        self.trees = {}
        for i in indexes:
            dname = "_".join([group_prefix + str(j) for j in i])
            full = os.path.join(path,dname)
            tree = KDTree(path=full,dtype=dtype,model_file=model_name,verbose=self.verbose,**kwargs)
            self.trees[dname] = tree  


    def __len__(self):
        return self.n

    def fit(self,X):    
        for i in self.indexes:
            dname = "_".join([self.group_prefix + str(j) for j in i])
            if self.trees[dname].tree is not None:
                if self.verbose:
                    print("INFO: Skip tree fit, model already existing - Change <path> in case of a new model!")
            else:
                if self.verbose:
                    print(f"INFO: model {dname} is trained")
                self.trees[dname].fit(X[:,i])


    '''
    ncached_idx - defines the number of indices are stored at once in memory
    '''

    def fit_seq(self,X_parts_list,parts_path=None,n_cached=1):
        assert len(self.trees) == len(self.indexes), "Error in initialization of trees - not fitting tree count"

        if parts_path is None:
            parts_path = self.path

        #Filter trained idxs
        idxs = []
        for i in self.indexes:
            dname = "_".join([self.group_prefix + str(j) for j in i])
            if self.trees[dname].tree is None:
                idxs.append(i)

        if len(idxs) > 0:
            c = 0
            while c < len(idxs):
                sub = idxs[c:c+n_cached]
                flat_idx = [item for sublist in sub for item in sublist] #Flatten list
                data = []
                for f in X_parts_list:
                    fname = os.path.join(parts_path,f)
                    x = np.load(fname)[:,flat_idx]
                    #x = torch.load(fname)[:,flat_idx].detach().numpy()
                    if x.dtype != np.dtype(self.dtype):
                        x = x.astype(self.dtype)
                    data.append(x)
                X = np.vstack(data)

                #Train models
                for i in range(len(sub)):
                    start = len([item for sublist in sub[:i] for item in sublist])
                    end = start+len(sub[i])
                    dname = dname = "_".join([self.group_prefix + str(j) for j in sub[i]])
                    if self.verbose:
                        print(f"INFO: Model {dname} is trained")
                    self.trees[dname].fit(X[:,start:end])
                c += n_cached

            self.trained = True
        else:
            if self.verbose:
                print("INFO: Skipping train as the model has already been trained! Change model_file in case of a new model!")

    def query(self,mins,maxs,idx):
        if mins.dtype != np.dtype(self.dtype):
            mins = mins.astype(self.dtype)
        if maxs.dtype != np.dtype(self.dtype):
            maxs = maxs.astype(self.dtype)

        #query stuff
        dname = "_".join([self.group_prefix + str(j) for j in idx])
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
            new_idx = np.arange(len(inds))
            if len(i_list) > 0:
                new_idx = np.where(np.in1d(inds,i_list) == False)
            i_list.extend(np.array(inds,dtype=np.int64)[new_idx])
            if no_pts == False:
                p_list.append(pts[new_idx])
        end = time.time()
        if self.verbose:
            print(f"INFO: query finished in {end-start} seconds")

        return i_list,p_list

    def compress_models(self,path=None,zipname="model"):
        if path is None:
            path = self.path
        for k,v in self.trees.items():
            zname = zipname + k + ".zip"
            v.compress_model(path,zname,folder=k)
            





