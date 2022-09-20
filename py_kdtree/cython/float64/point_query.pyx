# cython: profile=False
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport fabs,sqrt
        
cimport cython

cdef extern from "math.h":
    double INFINITY

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void recursive_search_point(double[::1] point,int k, double[:,:,::1] tree,int n_leaves,
                    int n_nodes,const double[:,:,::1] mmap,long[::1] indices_view,double[::1] distances_view):    
    cdef int i
    cdef long depth = 0 
    cdef long* indices = <long*> malloc(k * sizeof(long))
    cdef double* distances =   <double*> malloc(k * sizeof(double))

    try:
        #initializes distances
        for i in range(k):
            distances[i] = INFINITY

        indices,distances = _recursive_search_point(0,point,k,depth,tree,n_leaves,n_nodes,indices,distances,mmap)
        for i in range(k):
            indices_view[i] = indices[i]
            distances_view[i] = distances[i]
        return 
    finally:
        free(indices)   
        free(distances)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef (long*,double*) _recursive_search_point(int node_idx,double[::1] point,int k, int depth, double[:,:,::1] tree,int n_leaves, int n_nodes,
                          long* indices, double* distances,const double[:,:,::1] mmap) nogil:
    cdef int l_idx, r_idx,axis,first,second, lf_idx,dist_idx, i,j
    l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
    cdef double[:,:] l_bound,r_bound
    cdef double median, max_dist,max_dist_sub,dist,sub,power

    ############################## Leaf ##########################################################################
    if (l_idx >= tree.shape[0]) and (r_idx >= tree.shape[0]):
        #calculate distance for each contained point and check whether it is smaller than the ones found so far
        lf_idx = n_leaves+node_idx-n_nodes
        for j in range(mmap.shape[1]):
            if j == mmap.shape[1]-1:
                if mmap[lf_idx,j,0] == -1.:
                    continue
            dist = 0
            for i in range(tree.shape[1]): #dimensionality
                sub = mmap[lf_idx,j,i+1] - point[i]
                power = sub*sub
                dist = dist + power
            dist = sqrt(dist)
            max_dist = get_max(distances,k)
            if dist < max_dist:
                dist_idx = get_max_idx(distances,k)
                distances[dist_idx] = dist
                indices[dist_idx] = int(mmap[lf_idx,j,0])
        return indices,distances
    else:  
        axis = depth % tree.shape[1]

        median = tree[node_idx][axis][1]
        if point[axis] < median:
            first = l_idx
            second = r_idx
        else:
            first = r_idx
            second = l_idx
        indices,distances = _recursive_search_point(first,point,k,depth+1,tree,n_leaves,n_nodes,indices,distances,mmap)
        max_dist = get_max(distances,k)
        max_dist_sub = fabs(median - point[axis])
        if max_dist_sub < max_dist:
            indices,distances = _recursive_search_point(second,point,k,depth+1,tree,n_leaves,n_nodes,indices,distances,mmap)
        return indices,distances


@cython.boundscheck(False) 
@cython.wraparound(False)
cdef double get_max(double* array,int size) nogil:
    cdef int i
    cdef double MAX = -INFINITY
    for i in range(size):
        if array[i] > MAX:
            MAX = array[i]
    return MAX

@cython.boundscheck(False) 
@cython.wraparound(False)
cdef int get_max_idx(double* array,int size) nogil:
    cdef int i
    cdef double MAX = -INFINITY
    cdef int idx
    for i in range(size):
        if array[i] > MAX:
            MAX = array[i]
            idx = i
    return idx



    
    
