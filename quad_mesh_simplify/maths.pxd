
cdef void normal(double[:] v1, double[:] v2, double[:] v3, double[:] out)

cdef double norm(double[:] n)

cdef double dot1d(double[:] v1, double[:] v2)

cdef void calculate_K(double[:] p, double[:, :] K)

cdef void add_inplace(double[:, :] A, double[:, :] B)