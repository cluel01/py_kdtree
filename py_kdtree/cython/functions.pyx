from libc.stdlib cimport malloc, free, realloc
        
cimport cython
#from cython.parallel import prange
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef long[::1] recursive_search(double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves,
                    int n_nodes,const double[:,:,::1] mmap,double mem_cap):    
    cdef long[::1] indices_view
    cdef long ind_len = int(mmap.shape[0]*mmap.shape[1]*mem_cap) 
    cdef long extend_mem = ind_len

    cdef long ind_pt = 0 
    cdef long* indices = <long*> malloc(ind_len * sizeof(long))

    try:
        indices,ind_pt,ind_len = _recursive_search(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,0)
        indices_view = np.empty(ind_pt,dtype=np.int64)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        return indices_view 
    finally:
        free(indices)   
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*,long,long) _recursive_search(int node_idx,double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, long ind_pt,long ind_len,const double[:,:,::1] mmap,long extend_mem, int contained) nogil:
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
                    indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
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
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices,ind_pt,ind_len
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,1)
            indices,ind_pt,ind_len = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,0)
            
    return indices,ind_pt,ind_len

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int check_intersect(double[:,:] bounds,double[:] mins,double[:] maxs) nogil:
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
cdef int check_contained(double[:,:] bounds,double[:] mins,double[:] maxs) nogil:
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


#TODO so far contains both solutions malloc and realloc -> remove one of them
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef long* resize_long_array(long* arr,long old_len, long new_len) nogil:
    cdef long i 
    cdef long* mem = <long*> realloc(arr,new_len * sizeof(long))
    #cdef long* mem = <long*> malloc(new_len * sizeof(long))
    #if not mem:
    #    raise MemoryError()
    #for i in range(old_len):
    #    mem[i] = arr[i]
    arr = mem
    return arr
