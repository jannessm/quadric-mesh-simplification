import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

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
    cdef np.ndarray[DTYPE_LONG_T, ndim=2] edges
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] p, n
    
    cdef double[:, :] K_view
    cdef double[:] pos1, pos2, pos3, p_view, n_view
    
    cdef int i, j, a, num_nodes
    cdef long [:,:] edges_view
    
    num_nodes = positions.shape[0]

    edges = np.zeros((num_nodes, num_nodes), dtype=DTYPE_LONG) - 1
    edges_view = edges

    # create edges
    for i in range(face.shape[0]):
        for j in range(3):
            a = (j + 1) % 3
            if edges[face[i, j], face[i, a]] > -1:
                edges[face[i, j], face[i, a]] = -1
                edges[face[i, a], face[i, j]] = -1
            else:
                edges[face[i, j], face[i, a]] = i
                edges[face[i, a], face[i, j]] = i

    K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
    K_view = K
    
    p = np.zeros((4), dtype=DTYPE_DOUBLE)
    p_view = p

    n = np.zeros((4), dtype=DTYPE_DOUBLE)
    n_view = p

    for i in range(num_nodes):
        for j in range(num_nodes):
            if edges[i, j] < 0:
                continue
        
            # calculate face normal
            pos1 = positions[face[i, 0]]
            pos2 = positions[face[i, 1]]
            pos3 = positions[face[i, 2]]

            normal(pos1, pos2, pos3, n_view)

            # calculate penalties
            # calculate normal
            normal(pos1, pos2, n_view, p_view)
            calculate_K(p_view, K_view)
            mul_scalar_2D(K_view, 10e6)
            add_inplace(Q[i], K_view)
            add_inplace(Q[j], K_view)
