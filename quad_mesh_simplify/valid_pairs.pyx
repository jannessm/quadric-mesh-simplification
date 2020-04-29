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
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] distance
    cdef np.ndarray[DTYPE_LONG_T, ndim=2] valid_pairs, edges, pairs
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] f
    cdef long i

    valid_pairs = np.zeros((0,2), dtype=DTYPE_LONG)

    # option 1 to be valid
    for f in face:
        edges = np.array([
            [f[0], f[1]],
            [f[1], f[2]],
            [f[2], f[0]]
        ], dtype=DTYPE_LONG)

        valid_pairs = np.vstack([valid_pairs, edges])
        valid_pairs = remove_duplicates(valid_pairs)

    # option 2: distance below threshold
    
    if threshold is not None:
        for i in range(positions.shape[0]):
            distance = np.linalg.norm(positions - positions[i], axis=1)
            
            pairs = get_rows(distance < threshold)[:, None]
            # remove self-loops
            pairs = pairs[get_rows(pairs != i)]
            pairs = np.insert(pairs, 1, i, axis=1)

            valid_pairs = np.vstack([valid_pairs, pairs])
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