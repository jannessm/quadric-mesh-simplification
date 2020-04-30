from cpython cimport array

cdef class PairHeap:

    cdef array.array nodes_
    cdef public unsigned int [:] nodes
    cdef long length_
    cdef double[:, :] pairs

    cpdef long length(self)

    cdef double _get_value(self, long i)

    cdef double get_value(self, long i)
    
    cpdef double[:] get_pair(self, long i)

    cdef void insert(self, long i)

    cdef void build(self)

    cdef double[:] pop(self)

    cdef void percolate_up(self, long i)

    cdef void percolate_down(self, long i)

    cdef long get_min_child(self, long i)


