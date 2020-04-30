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
cpdef void update_pairs(
    long v1,
    long v2,
    PairHeap heap,
    double [:, :] positions,
    double [:, :, :] Q,
    double [:, :] features):
    """updates all pairs by calculating all errors according to the contracted edge and removes the contracted
    pair from the pairs array.

    Args:
        pairs (:class:`ndarray`): list of pairs
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        Q (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        features (:class:`ndarray`, optional): array of shape num_nodes x num_features

    :rtype: :class:`ndarray`
    """
    cdef long i
    cdef double[:] pair

    # iterate over all pairs and recalculate error if needed
    for i in range(heap.length()):
        pair = heap.get_pair(i)
        if (pair[1] == v1 or pair[2] == v1) and \
            (pair[1] == v2 or pair[2] == v2):
            pair[0] = -1 # low value so pair will appear first and gets unvalid in simplify.pyx
            continue

        if pair[1] == v1 or pair[1] == v2:
            calculate_pair_attributes(
                v1,
                <long>pair[2],
                positions,
                Q,
                features,
                pair)
        
        elif pair[2] == v1 or pair[2] == v2:
            calculate_pair_attributes(
                <long>pair[1],
                v1,
                positions,
                Q,
                features,
                pair)
    
    heap.build()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void update_face(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    list deleted_faces):
    """updates a face for a contracted pair by removing all faces accordingly.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        face (:class:`ndarray`): face array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef long v1, v2, i
    cdef list rows
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] rows_v1, rows_v2
    
    rows = []
    v1 = <long>pair[1]
    v2 = <long>pair[2]

    # update face from new pairs
    rows_v1 = get_rows(face == v1)
    rows_v2 = get_rows(face == v2)

    # 1. remove faces with both nodes of pair
    for i in rows_v1:
        if i in rows_v2:
            rows.append(i)
    #face = np.delete(face, rows, 0)
    deleted_faces = face + rows

    # 2. point faces to new merged node
    face[face == v2] = v1

    # update indexes for all vertices that where shifted in array
    #face[face > v2] -= 1

    #return face

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void update_features(
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
    #v2 = <long>pair[2]

    if features is not None:
        features[v1] = pair[feature_offset:]
        #features = np.delete(features, v2, 0)

