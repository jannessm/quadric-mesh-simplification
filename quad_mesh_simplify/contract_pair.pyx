import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

cimport cython
from .targets cimport calculate_pair_attributes
from .utils cimport get_rows

cdef int target_offset = 3
cdef int feature_offset = 3 + 3

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] update_positions(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions):
    """updates all positions by contracting the given pair. In detail, it sets the position for the first node to
    the optimal position of the contracted pair and removes the position of the second node.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        positions (:class:`ndarray`): positions array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef long v1, v2
    v1 = <long>pair[1]
    v2 = <long>pair[2]

    positions[v1] = np.copy(pair[target_offset:feature_offset])
    return np.delete(positions, v2, 0)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] update_Q(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q):
    """updates all Q matrixes by contracting the given pair. In detail, it sets the matrix for the first node to
    the sum of both Qs from the pair and removes the matrix of the second node.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        Q (:class:`ndarray`): Q array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef long v1, v2
    v1 = <long>pair[1]
    v2 = <long>pair[2]
    
    # update Q
    Q[v1] = Q[v1] + Q[v2]
    return np.delete(Q, v2, 0)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] update_pairs(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features):
    """updates all pairs by calculating all errors according to the contracted edge and removes the contracted
    pair from the pairs array.

    Args:
        pairs (:class:`ndarray`): list of pairs
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        Q (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        features (:class:`ndarray`, optional): array of shape num_nodes x num_features

    :rtype: :class:`ndarray`
    """
    cdef long v1, v2, i
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] rows, cols
    
    v1 = <long>pairs[0, 1]
    v2 = <long>pairs[0, 2]

    # set all v2 to v1 since it doesnt exists anymore and was replaced by the
    # new target
    rows, cols = np.where(pairs[:, 1:3] == v2)
    pairs[rows, cols + 1] = v1
    rows, cols = np.where(pairs[:, 1:3] > v2)
    pairs[rows, cols + 1] -= 1

    # update all rows with indexes of v1
    rows = get_rows(pairs[:, 1:3] == v1)
    for i in rows:
        p = pairs[i]
        if p[1] == v1:
            pairs[i] = calculate_pair_attributes(v1, p[2], positions, Q, features)
        elif p[2] == v1:
            pairs[i] = calculate_pair_attributes(p[1], v1, positions, Q, features)


    pairs = np.delete(pairs, 0, 0)
    return sort_by_error(pairs)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_LONG_T, ndim=2] update_face(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_LONG_T, ndim=2] face):
    """updates a face for a contracted pair by removing all faces accordingly.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        face (:class:`ndarray`): face array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef long v1, v2, i
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] rows, rows_v1, rows_v2
    
    v1 = <long>pair[1]
    v2 = <long>pair[2]

    # update face from new pairs
    rows_v1 = get_rows(face == v1)
    rows_v2 = get_rows(face == v2)
    rows = np.zeros((0), dtype=DTYPE_LONG)

    # 1. remove faces with both nodes of pair
    for i in rows_v1:
        if i in rows_v2:
            rows = np.append(rows, i)
    face = np.delete(face, rows, 0)

    # 2. point faces to new merged node
    face[face == v2] = v1

    # update indexes for all vertices that where shifted in array
    face[face > v2] -= 1

    return face

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] update_features(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features):
    """updates all features by contracting the given pair. In detail, it sets the features for the first node to
    the feature vector of the contracted pair and removes the features of the second node.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        features (:class:`ndarray`): features array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef long v1, v2
    v1 = <long>pair[1]
    v2 = <long>pair[2]

    if features is not None:
        features[v1] = pair[feature_offset:]
        features = np.delete(features, v2, 0)

    return features

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] sort_by_error(np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs):
    """sort a pairs array by the first column

    Args:
        pairs (:class:`ndarray`): array that will be sorted
    
    :rtype: :class:`ndarray`
    """
    cdef view = ', '.join(['double' for _ in range(pairs.shape[1])])
    
    if pairs.shape[0] > 0:
        pairs.view(view).sort(order=['f0'], axis=0)
        return np.unique(pairs, axis=0)
    else:
        return pairs
