cpdef long[::1] recursive_search(double[::1] ,double[::1] , double[:,:,::1] ,int ,
                    int ,const double[:,:,::1] ,double mem_cap) 

cdef (long*,long,long) _recursive_search(int ,double[::1] ,double[::1] , double[:,:,::1] ,int , int ,
                          long* , int ,long ,const double[:,:,::1] ,long ,int ) nogil

cdef int check_intersect(double[:,:] ,double[:] ,double[:] ) nogil

cdef int check_contained(double[:,:] ,double[:] ,double[:] ) nogil

cdef long* resize_long_array(long* ,long, long ) nogil