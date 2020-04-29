import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

from .utils cimport get_faces_for_node, face_normal

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] compute_Q(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face):
    r"""computes for all nodes in :obj:`positions` a 4 x 4 matrix Q used for calculating error values.

    The error is later calculated by (v.T Q v) and forms the quadric error metric.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face

    :rtype: :class:`ndarray`"""

    assert(face.shape[1] == 3)
    assert(positions.shape[1] == 3)

    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] f
    cdef double d
    #cdef long num_nodes, i
    cdef long num_nodes, i

    num_nodes = positions.shape[0]
    Q = np.zeros((num_nodes, 4, 4), dtype=DTYPE_DOUBLE)

    for f in face:
        Q[f] += get_K(positions[f])

    return Q

cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] get_K(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions):
    
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] u, n
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] p
    cdef double d
    
    u = positions[0]
    n = face_normal(positions, True, -1)
    d = - n.dot(u)
    p = np.hstack([n,d])[:, None]
    
    return p.dot(p.T)