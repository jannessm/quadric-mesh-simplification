import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long
DTYPE_INT = np.int8

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T
ctypedef np.int8_t DTYPE_INT_T

from .maths cimport normal, calculate_K, add_inplace, mul_scalar_2D
from cpython cimport array
import array
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void preserve_bounds(
    double [:, :] positions,
    long [:, :] face,
    double [:, :, :] Q):
    """This method adds a large penality to the current Q matrix for each node of edge that is only part of one face and therefore forms a boundary.

    Note that it manipulates the matrix Q in place.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        Q (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
    """
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] K
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] p, n
    cdef np.ndarray[DTYPE_INT_T, ndim=2] edges_
    
    cdef double[:, :] K_view
    cdef double[:] pos1, pos2, pos3, p_view, n_view
    
    cdef int i, j, a, num_nodes, v1, v2
    cdef char [:,:] edges
    
    num_nodes = positions.shape[0]

    edges_ = np.zeros((num_nodes, num_nodes), dtype=DTYPE_INT) - 1
    edges = edges_

    # create edges
    for i in range(face.shape[0]):
        for j in range(3):
            a = (j + 1) % 3
            v1 = face[i, j]
            v2 = face[i, a]
            if edges[v1, v2] > -1:
                edges[v1, v2] = -2
                edges[v2, v1] = -2
            elif edges[v1, v2] > -2:
                edges[v1, v2] = i
                edges[v2, v1] = i

    K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
    K_view = K
    
    p = np.zeros((4), dtype=DTYPE_DOUBLE)
    p_view = p

    n = np.zeros((4), dtype=DTYPE_DOUBLE)
    n_view = p

    # add penalities
    for i in range(face.shape[0]):
        for j in range(3):
            a = (j + 1) % 3
            v1 = face[i, j]
            v2 = face[i, a]
            
            if edges[v1, v2] > -1:

                # calculate face normal
                pos1 = positions[v1]
                pos2 = positions[v2]
                pos3 = positions[face[i, (j + 2) % 3]]

                normal(pos1, pos2, pos3, n_view)

                # calculate penalties
                # calculate normal
                normal(pos1, pos2, n_view, p_view)
                calculate_K(p_view, K_view)
                mul_scalar_2D(K_view, 10e3)
                add_inplace(Q[v1], K_view)
                add_inplace(Q[v2], K_view)
