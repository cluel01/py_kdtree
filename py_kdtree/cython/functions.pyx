@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
cdef recursive_search(int node_idx,double[:,:] mins,double[:,:] maxs, double[:,:] tree, int depth,
                    *int leaf_idxs int leaf_pt) nogil:
    #Idea: only traverse the leaf idxs and then do the comparison afterwards in numpy 
    #-> also include the fully contain comparison during traversal -> then include all leaf idxs of this branch
        l_idx,r_idx = (2*node_idx)+1, (2*node_idx)+2
        
        if (l_idx >= depth) and (r_idx >= depth):
            #TODO add leaf_idx to leaf_idxs array -> how to deal with the pointer?

        l_bounds = tree[l_idx]
        r_bounds = tree[r_idx]

        #########
        #fully contained
        if (np.all(bounds[:,0] >= mins )) and (np.all(maxs >= bounds[:,1])):
            #TODO generate all possible leaf ids up to depth of tree

        #if at least intersects
        if (np.all(l_bounds[:,1] >= mins )) and (np.all(maxs >= l_bounds[:,0])):
            self._recursive_search(l_idx,mins,maxs,indices,points)

        if (np.all(r_bounds[:,1] >= mins )) and (np.all(maxs >= r_bounds[:,0])):
            self._recursive_search(r_idx,mins,maxs,indices,points)

        return indices,points