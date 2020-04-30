
cdef void normal(double[:] v1, double[:] v2, double[:] v3, double[:] out)

cdef double norm(double[:] n)

cdef double dot1d(double[:] v1, double[:] v2)

cdef void calculate_K(double[:] p, double[:, :] K)

cdef void add_inplace(double[:, :] A, double[:, :] B)

cdef void add_2D(double[:, :] A, double[:, :] B, double [:, :] R)

cdef void mul_scalar_1D(double[:] a, double scalar)

cdef void mul_scalar_2D(double[:, :] A, double scalar)

cdef double error(double[:] p, double[:, :] Q)