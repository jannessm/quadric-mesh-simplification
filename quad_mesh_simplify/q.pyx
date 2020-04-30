import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

cimport cython

from .maths cimport normal, calculate_K, add_inplace

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] compute_Q(
    double [:, :] positions,
    long [:, :] face):
    r"""computes for all nodes in :obj:`positions` a 4 x 4 matrix Q used for calculating error values.

    The error is later calculated by (v.T Q v) and forms the quadric error metric.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face

    :rtype: :class:`ndarray`"""

    assert(face.shape[1] == 3)
    assert(positions.shape[1] == 3)

    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] K
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] p
    cdef double d, n
    cdef double[:, :, :] Q_view
    cdef double[:, :] K_view
    cdef double[:] pos1, pos2, pos3, p_view
    cdef long num_nodes, i, j


    num_nodes = positions.shape[0]
    
    Q = np.zeros((num_nodes, 4, 4), dtype=DTYPE_DOUBLE)
    Q_view = Q

    K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
    K_view = K
    
    p = np.zeros((4), dtype=DTYPE_DOUBLE)
    p_view = p

    for i in range(face.shape[0]):
        pos1 = positions[face[i, 0]]
        pos2 = positions[face[i, 1]]
        pos3 = positions[face[i, 2]]

        normal(pos1, pos2, pos3, p_view)
        calculate_K(p_view, K_view)
        add_inplace(Q_view[face[i, 0]], K_view)
        add_inplace(Q_view[face[i, 1]], K_view)
        add_inplace(Q_view[face[i, 2]], K_view)

    return Q
