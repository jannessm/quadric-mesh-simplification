
cimport cython

cdef extern from "math.h" nogil:
  double sqrt(double x)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void normal(double[:] v1, double[:] v2, double[:] v3, double[:] out) nogil:
    cdef int i, a, b
    cdef double n
    for i in range(3):
        a = (i + 1) % 3
        b = (i + 2) % 3
        out[i] = (v1[a] - v2[a]) * (v3[b] - v2[b]) - \
                 (v1[b] - v2[b]) * (v3[a] - v2[a])
    out[3] = 0
    n = norm(out)
    if n > 0:
        for i in range(3):
            out[i] /= n
    
    # set d
    out[3] = - dot1d(out, v1)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double norm(double[:] n) nogil:
    cdef double s = 0.
    cdef int i
    for i in range(n.shape[0]):
        s += n[i] * n[i]
    return sqrt(s)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double dot1d(double[:] v1, double[:] v2) nogil:
    cdef double s = 0.
    cdef int i
    for i in range(v1.shape[0]):
        s += v1[i] * v2[i]
    return s

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void calculate_K(double[:] p, double[:, :] K):
    cdef double s = 0.
    cdef int i, j
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            K[i ,j] = p[i] * p[j]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void add_inplace(double[:, :] A, double[:, :] B):
    cdef int i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] += B[i,j]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void add_2D(double[:, :] A, double[:, :] B, double [:, :] R):
    cdef int i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            R[i, j] = A[i,j] + B[i,j]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void mul_scalar_1D(double[:] a, double scalar):
    cdef int i
    for i in range(a.shape[0]):
        a[i] *= scalar

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void mul_scalar_2D(double[:, :] A, double scalar):
    cdef int i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] *= scalar

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double error(double[:] p, double[:, :] Q):
    """p (shape (3))
    Q (shape (4,4))"""
    cdef int i, j
    cdef double error
    error = 0
    
    for i in range(Q.shape[1] - 1):
        for j in range(p.shape[0]):
            error += p[j] * Q[j, i] * p[i]
        error += Q[3, i] * p[i]

    for j in range(p.shape[0]):
        error += p[j] * Q[j, 3]
    error += Q[3, 3]
    
    return error
