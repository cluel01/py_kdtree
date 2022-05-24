cpdef long[::1] recursive_search(float[::1] ,float[::1] , float[:,:,::1] ,int ,
                    int ,const float[:,:,::1] ,int ,int ,double ,int[::1] ) 

cdef (long*,long,long,int) _recursive_search(int ,float[::1] ,float[::1] , float[:,:,::1] ,int , int ,
                          long* , long ,long ,const float[:,:,::1] ,long ,int,int ) nogil

cdef (long*,long,long,int) _recursive_search_limit(int ,float[::1] ,float[::1] , float[:,:,::1] ,int , int ,
                          long* , long ,long ,const float[:,:,::1] ,long ,int ,int,int ) nogil

cdef (long*,long,long,int) _recursive_search_limit_leaves(int ,float[::1] ,float[::1] , float[:,:,::1] ,int , int ,
                          long* , long ,long ,const float[:,:,::1] ,long ,int ,int,int ) nogil

cdef int check_intersect(float[:,:] ,float[:] ,float[:] ) nogil

cdef int check_contained(float[:,:] ,float[:] ,float[:] ) nogil

cdef long* resize_long_array(long* ,long, long ) nogil

cpdef long[::1] recursive_search_time(float[::1] ,float[::1] , float[:,:,::1] ,int ,
                    int ,const float[:,:,::1] ,int,float ,float[::1],int[::1])

cdef (int,int) _recursive_search_time(int ,float[::1] ,float[::1] , float[:,:,::1] ,int , int ,
                         int ,int* , int , int* , int ) nogil

cdef (long,float,float,long*,int,int) _filter_leaves(const float[:,:,::1] ,float[::1],float[::1],long* ,long ,int* ,int ,int* ,int ,
                                        long ,int, long ) nogil
