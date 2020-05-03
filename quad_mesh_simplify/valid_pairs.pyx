import numpy as np
cimport numpy as np
DTYPE_CHAR = np.dtype('B')
DTYPE_LONG = np.long
DTYPE_DOUBLE = np.double

from .maths cimport norm

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_LONG_T, ndim=2] compute_valid_pairs(
    double [:, :] positions,
    long [:, :] face,
    double threshold):
    """computes all valid pairs. A valid pair of nodes is either nodes that are connected by an edge, or their distance are below the given threshold.
    The returned array is of shape (num_pairs, 2) containing the ids of all nodes.

    Args:
        positions (:class:`ndarray`): positions of nodes with shape (num_nodes, 3)
        face (:class:`ndarray`): face of mesh containing node ids with shape (num_faces, 3)
        threshold (:class:`double`, optional): if provided, includes pairs that have a smaller distance than this threshold. (default: :obj:`0`)

    :rtype: :class:`ndarray`
    """
    cdef np.ndarray[DTYPE_CHAR_T, ndim=2] adj_matrix_
    cdef unsigned char [:, :] adj_matrix
    cdef double [:] distance
    cdef int i, j, num_nodes, v1, v2, v3

    num_nodes = positions.shape[0]

    adj_matrix_ = np.zeros((num_nodes, num_nodes), dtype=DTYPE_CHAR)
    adj_matrix = adj_matrix_

    distance_ = np.zeros((3), dtype=DTYPE_DOUBLE)
    distance = distance_

    # option 1: connected by an edge
    for i in range(face.shape[0]):
        for j in range(3):
            v1 = face[i, j]
            v2 = face[i, (j + 1) % 3]
            v3 = face[i, (j + 2) % 3]
            if not adj_matrix[v1, v2]:
                adj_matrix[v1, v2] = True
                adj_matrix[v2, v1] = True
            if not adj_matrix[v2, v3]:
                adj_matrix[v2, v3] = True
                adj_matrix[v3, v2] = True
            if not adj_matrix[v1, v3]:
                adj_matrix[v1, v3] = True
                adj_matrix[v3, v1] = True

    # option 2: distance below threshold
    if threshold > 0.:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes): # self loops not allowed
                if adj_matrix[i, j]:
                    continue

                # else check distance
                for k in range(3):
                    distance[k] = positions[i, k] - positions[j, k]

                if norm(distance) < threshold:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True

    adj_matrix = np.triu(adj_matrix)
    return np.vstack(np.where(adj_matrix)).T.astype(DTYPE_LONG)
