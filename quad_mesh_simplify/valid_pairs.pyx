import numpy as np

DTYPE_LONG = np.int64

from .utils cimport get_faces_for_node
from .utils cimport get_rows

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] compute_valid_pairs(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    double threshold):
    """computes all valid pairs. A valid pair of nodes is either nodes that are connected by an edge, or their distance are below the given threshold.
    The returned array is of shape (num_pairs, 2) containing the ids of all nodes.

    Args:
        positions (:class:`ndarray`): positions of nodes with shape (num_nodes, 3)
        face (:class:`ndarray`): face of mesh containing node ids with shape (num_faces, 3)
        threshold (:class:`double`, optional): if provided, includes pairs that have a smaller distance than this threshold. (default: :obj:`0`)

    :rtype: :class:`ndarray`
    """
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] distances
    cdef np.ndarray[DTYPE_LONG_T, ndim=2] valid_pairs
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] rows, cols
    cdef int i, j, num_nodes

    num_nodes = positions.shape[0]

    valid_pairs = np.zeros((0,2), dtype=DTYPE_LONG)

    # option 1: connected by an edge
    valid_pairs = np.vstack([
        face[:, :2],
        face[:, 1:],
        face[:, [0,2]]
    ]).astype(DTYPE_LONG)

    # option 2: distance below threshold
    if threshold > 0.:
        distances = np.zeros((num_nodes, num_nodes)) + np.eye(num_nodes) * threshold
        
        for i in range(num_nodes):
            distances[:, i] += np.linalg.norm(positions - positions[i], axis=1)
        
        rows, cols = np.where(distances < threshold)
        
        valid_pairs = np.vstack([
            valid_pairs,
            np.vstack([rows, cols]).T
        ])

    valid_pairs = remove_duplicates(valid_pairs)
    return valid_pairs

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef np.ndarray[DTYPE_LONG_T, ndim=2] remove_duplicates(#
    np.ndarray[DTYPE_LONG_T, ndim=2] arr):
    """removes duplicates in an array of edges.
    
    Args:
        arr (:class:`ndarray`): array

    :rtype: :class:`ndarray`"""
    return np.unique(np.sort(arr, axis=1), axis=0)