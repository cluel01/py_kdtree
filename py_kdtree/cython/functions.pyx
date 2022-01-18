# cython: profile=False
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdlib cimport malloc, free, realloc
        
cimport cython
#from cython.parallel import prange
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef long[::1] recursive_search(double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves,
                    int n_nodes,const double[:,:,::1] mmap,int max_pts,double mem_cap,int[::1] arr_loaded):    
    cdef long[::1] indices_view
    cdef long ind_len = int(mmap.shape[0]*mmap.shape[1]*mem_cap) 
    cdef long extend_mem = ind_len

    cdef long ind_pt = 0 
    cdef long* indices = <long*> malloc(ind_len * sizeof(long))

    cdef int loaded_leaves = 0

    try:
        if max_pts > 0:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,0)
        else:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search(0,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)
        arr_loaded[0] = loaded_leaves
        indices_view = np.empty(ind_pt,dtype=np.int64)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        return indices_view 
    finally:
        free(indices)   

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*,long,long,int) _recursive_search(int node_idx,double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, long ind_pt,long ind_len,const double[:,:,::1] mmap,long extend_mem, int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx,intersects, ret,lf_idx,isin,j,k
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef double[:,:] bounds,l_bounds,r_bounds
    cdef double leaf_val
    
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        loaded_leaves += 1
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
                    leaf_val = mmap[lf_idx,j,k+1]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(mmap[lf_idx,j,0])
                    ind_pt += 1
                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,loaded_leaves,0)
            
    return indices,ind_pt,ind_len,loaded_leaves

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long*,long,long,int) _recursive_search_limit(int node_idx,double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, long ind_pt,long ind_len,const double[:,:,::1] mmap,long extend_mem, int max_pts,int loaded_leaves, int contained) nogil:
    cdef int l_idx, r_idx,intersects, ret,lf_idx,isin,j,k
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef double[:,:] bounds,l_bounds,r_bounds
    cdef double leaf_val
    
    if ind_pt == max_pts:
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        loaded_leaves += 1
        if contained == 1:
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if mmap[lf_idx,j,0] == -1.:
                        continue
                indices[ind_pt] = int(mmap[lf_idx,j,0])
                ind_pt += 1
                if ind_pt == max_pts:
                    return indices,ind_pt,ind_len,loaded_leaves
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
                    leaf_val = mmap[lf_idx,j,k+1]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(mmap[lf_idx,j,0])
                    ind_pt += 1

                    if ind_pt == max_pts:
                        return indices,ind_pt,ind_len,loaded_leaves

                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        return indices,ind_pt,ind_len,loaded_leaves
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
            indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(l_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,0)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,1)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    indices,ind_pt,ind_len,loaded_leaves = _recursive_search_limit(r_idx,mins,maxs,tree,n_leaves,n_nodes,indices,ind_pt,ind_len,mmap,extend_mem,max_pts,loaded_leaves,0)
            
    return indices,ind_pt,ind_len,loaded_leaves

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


################################################### Profiling variant of KDtree  #########################################

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef long[::1] recursive_search_time(double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves,
                    int n_nodes,const double[:,:,::1] mmap,int max_pts,double mem_cap,double[::1] times, int[::1] loaded_leaves):    
    cdef long[::1] indices_view
    cdef long ind_len = int(mmap.shape[0]*mmap.shape[1]*mem_cap) 
    cdef long extend_mem = ind_len

    cdef long ind_pt = 0 
    cdef long* indices = <long*> malloc(ind_len * sizeof(long))

    cdef int pt_is = 0
    cdef int pt_ct = 0
    cdef int* leaves_intersected = <int*> malloc(n_leaves * sizeof(int))
    cdef int* leaves_contained = <int*> malloc(n_leaves * sizeof(int))

    cdef double* leaf = <double*> malloc(mmap.shape[1]*mmap.shape[2] * sizeof(double))
    
    cdef timespec ts
    cdef double loading_time,filter_time,total_time

    if max_pts <= 0:
        max_pts = mmap.shape[0]*mmap.shape[1]

    try:
        clock_gettime(CLOCK_REALTIME, &ts)
        start = ts.tv_sec + (ts.tv_nsec / 1000000000.) 
        pt_is,pt_ct = _recursive_search_time(0,mins,maxs,tree,n_leaves,n_nodes,0,leaves_intersected,pt_is,leaves_contained,pt_ct)
        ind_pt,loading_time,filter_time,indices,pt_is,pt_ct = _filter_leaves(leaf,mmap,mins,maxs,indices,ind_pt,leaves_intersected,pt_is,leaves_contained,pt_ct,ind_len,max_pts,extend_mem)
        indices_view = np.empty(ind_pt,dtype=np.int64)
        for i in range(ind_pt):
            indices_view[i] = indices[i]
        clock_gettime(CLOCK_REALTIME, &ts)
        end = ts.tv_sec + (ts.tv_nsec / 1000000000.)

        total_time = end-start
        times[0] = total_time
        times[1] = loading_time
        times[2] = filter_time

        loaded_leaves[0] = pt_ct
        loaded_leaves[1] = pt_is

        return indices_view
    finally:
        free(indices)   
        free(leaves_intersected)
        free(leaves_contained)  
        free(leaf)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (int,int) _recursive_search_time(int node_idx,double[::1] mins,double[::1] maxs, double[:,:,::1] tree,int n_leaves, int n_nodes,
                         int contained,int* leaves_intersected, int pt_is, int* leaves_contained, int pt_ct) nogil:
    cdef int l_idx, r_idx,intersects, ret,lf_idx
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef double[:,:] bounds,l_bounds,r_bounds
    
    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        lf_idx = n_leaves+node_idx-n_nodes
        if contained == 1:
            leaves_contained[pt_ct] = lf_idx
            pt_ct += 1
        else:
            leaves_intersected[pt_is] = lf_idx
            pt_is += 1
        return pt_is,pt_ct
    ############################## Normal node ##########################################################################
    else:
        if contained == 1:
            pt_is,pt_ct = _recursive_search_time(l_idx,mins,maxs,tree,n_leaves,n_nodes,1,leaves_intersected, pt_is, leaves_contained, pt_ct)
            pt_is,pt_ct = _recursive_search_time(r_idx,mins,maxs,tree,n_leaves,n_nodes,1,leaves_intersected, pt_is, leaves_contained, pt_ct)
        else:
            l_bounds = tree[l_idx]
            r_bounds = tree[r_idx]
            ret = check_contained(l_bounds,mins,maxs)
            if ret == 1:
                pt_is,pt_ct = _recursive_search_time(l_idx,mins,maxs,tree,n_leaves,n_nodes,1,leaves_intersected, pt_is, leaves_contained, pt_ct)
            else:
                ret = check_intersect(l_bounds,mins,maxs)
                if ret == 1:
                    pt_is,pt_ct = _recursive_search_time(l_idx,mins,maxs,tree,n_leaves,n_nodes,0,leaves_intersected, pt_is, leaves_contained, pt_ct)

            ret = check_contained(r_bounds,mins,maxs)
            if ret == 1:
                pt_is,pt_ct = _recursive_search_time(r_idx,mins,maxs,tree,n_leaves,n_nodes,1,leaves_intersected, pt_is, leaves_contained, pt_ct)
            else:
                ret = check_intersect(r_bounds,mins,maxs)
                if ret == 1:
                    pt_is,pt_ct = _recursive_search_time(r_idx,mins,maxs,tree,n_leaves,n_nodes,0,leaves_intersected, pt_is, leaves_contained, pt_ct)
            
    return pt_is,pt_ct

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (long,double,double,long*,int,int) _filter_leaves(double* leaf,const double[:,:,::1] mmap,double[::1] mins,double[::1] maxs,long* indices,long ind_pt,int* leaves_intersected,int pt_is,int* leaves_contained,int pt_ct,
                                        long ind_len,int max_pts, long extend_mem) nogil:
    cdef int i,j,k,isin,contained,leaf_pt,pt_ct_real, pt_is_real 
    cdef double leaf_val,start,end,loading_time,filter_time 
    cdef timespec ts
    pt_ct_real, pt_is_real = 0,0
    loading_time , filter_time = 0,0
    
    for i in range(pt_is+pt_ct):
        if i >= pt_is:
            leaf_idx = leaves_contained[i-pt_is]
            contained = 1
        else:
            leaf_idx = leaves_intersected[i]
            contained = 0

        clock_gettime(CLOCK_REALTIME, &ts)
        start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
        # Load leaf into memory
        if contained:
            for j in range(mmap.shape[1]):
                leaf[j] = mmap[leaf_idx,j,0]
                if j == mmap.shape[1]-1:
                    if leaf[j] == -1.:
                        leaf_pt = mmap.shape[1]-1
                    else:
                        leaf_pt = mmap.shape[1]
            pt_ct_real += 1
        else:
            leaf_pt = 0
            for j in range(mmap.shape[1]):
                if j == mmap.shape[1]-1:
                    if mmap[leaf_idx,j,0] == -1.:
                        continue

                for k in range(mmap.shape[2]):
                    leaf[leaf_pt] = mmap[leaf_idx,j,k]
                    leaf_pt += 1
            pt_is_real += 1
        clock_gettime(CLOCK_REALTIME, &ts)
        end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
        loading_time += end-start

        #Filter
        clock_gettime(CLOCK_REALTIME, &ts)
        start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
        if contained == 1:
            for j in range(leaf_pt):
                indices[ind_pt] = int(leaf[j])
                ind_pt += 1

                if ind_pt == max_pts:
                    clock_gettime(CLOCK_REALTIME, &ts)
                    end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
                    filter_time += end-start

                    return ind_pt,loading_time,filter_time,indices,pt_is_real,pt_ct_real

                if ind_pt == ind_len:
                    indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                    ind_len += extend_mem
        else:
            for j from 0 <= j < leaf_pt by mmap.shape[2]:
                k = 0
                isin = 0
                while (k < mmap.shape[2]-1) and (isin == k):
                    leaf_val = leaf[j+k+1]#leaf[j+k+1]
                    if (leaf_val >= mins[k]) and (leaf_val <= maxs[k]):
                        isin += 1
                    k += 1
                if isin == k:
                    indices[ind_pt] = int(leaf[j])
                    ind_pt += 1

                    if ind_pt == max_pts:
                        clock_gettime(CLOCK_REALTIME, &ts)
                        end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
                        filter_time += end-start

                        return ind_pt,loading_time,filter_time,indices,pt_is_real,pt_ct_real

                    if ind_pt == ind_len:
                        indices = resize_long_array(indices,ind_len,ind_len+extend_mem)
                        ind_len += extend_mem
        clock_gettime(CLOCK_REALTIME, &ts)
        end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
        filter_time += end-start

    return ind_pt,loading_time,filter_time,indices,pt_is_real,pt_ct_real