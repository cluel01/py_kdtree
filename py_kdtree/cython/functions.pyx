from libc.stdlib cimport malloc, free, realloc
        
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef long[::1] recursive_search(double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves,
                    int n_nodes,const double[:,:,::1] mmap):    
    cdef long[::1] indices_view
    cdef long ind_len = int(mmap.shape[0]*mmap.shape[1]*0.0001) #PARAMETER

    cdef int ind_pt = 0 
    cdef long* indices = <long*> malloc(ind_len * sizeof(long))
    
    try:
        ind_pt = _recursive_search(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,0)
        indices_view = np.empty(ind_pt,dtype=np.int64)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        return indices_view
    finally:
        free(indices)
    
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int _recursive_search(int node_idx,double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, int ind_pt,long ind_len,const double[:,:,::1] mmap,int contained):
    cdef int l_idx, r_idx,intersects, ret,lf_idx,isin,j,k
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef double[:,:] bounds,l_bounds,r_bounds
    
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if mmap[lf_idx,j,0] == -1.:
                        continue
                indices[ind_pt] = int(mmap[lf_idx,j,0])
                ind_pt += 1
                if ind_pt == ind_len:
                    resize_long_array(indices,ind_len+mmap.shape[1])
                    ind_len += mmap.shape[1]
        else:
            for j in range(mmap.shape[1]):
                k = 0
                isin = 0
                while (k < mmap.shape[2]-1) and (isin == k):
                    if j == mmap.shape[1]-1:
                        if mmap[lf_idx,j,0] == -1.:
                            k += 1
                            continue
                    if (mmap[lf_idx,j,k+1] >= mins[k]) and (mmap[lf_idx,j,k+1] <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(mmap[lf_idx,j,0])
                    ind_pt += 1
                    if ind_pt == ind_len:
                        resize_long_array(indices,ind_len+mmap.shape[1])
                        ind_len += mmap.shape[1]
        return ind_pt
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            ind_pt = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,1)
            ind_pt = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                ind_pt = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    ind_pt = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                ind_pt = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    ind_pt = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,0)
            
    return ind_pt

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int check_intersect(double[:,:] bounds,double[:] mins,double[:] maxs):
    cdef int intersects, idx
    
    intersects = 0
    idx = 0
    while (idx < bounds.shape[0]) and (intersects == idx):
        if (bounds[idx,1] >= mins[idx]) and (bounds[idx,0] <= maxs[idx]):
            intersects += 1
        idx += 1
    
    if intersects == idx:
        intersects = 1
    else:
        intersects = 0
    
    return intersects

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int check_contained(double[:,:] bounds,double[:] mins,double[:] maxs):
    cdef int contained, idx
    
    contained = 0
    idx = 0
    while (idx < bounds.shape[0]) and (contained == idx):
        if (bounds[idx,0] >= mins[idx]) and (bounds[idx,1] <= maxs[idx]):
            contained += 1
        idx += 1
        
    if contained == idx:
        contained = 1
    else:
        contained = 0
    
    return contained

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef long* resize_long_array(long* arr, long new_len):
    mem = <long*> realloc(arr, new_len * sizeof(long))
    #if not mem:
    #    raise MemoryError()
    arr = mem
    return arr
