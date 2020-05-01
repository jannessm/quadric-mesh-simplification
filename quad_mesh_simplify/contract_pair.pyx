import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

cimport cython
from .targets cimport calculate_pair_attributes
from .utils cimport get_rows

import array

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
            pair[0] = -10e6 # low value so pair will appear first and gets unvalid in simplify.pyx
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
    long v1,
    long v2,
    long [:, :] face,
    unsigned char [:] deleted_faces):
    """updates a face for a contracted pair by removing all faces accordingly.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        face (:class:`ndarray`): face array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef int i, j, v1_in_face, v2_in_face

    for i in range(face.shape[0]):
        if deleted_faces[i]:
            continue

        v1_in_face = False
        v2_in_face = False
        
        for j in range(3):
            if v1 == face[i, j]:
                v1_in_face = True
            if v2 == face[i, j]:
                v2_in_face = True
                
                # 2. point faces to new merged node
                face[i, j] = v1

        # 1. remove faces with both nodes of pair
        if v1_in_face and v2_in_face:
            deleted_faces[i] = True


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void update_features(
    double [:] pair,
    double [:, :] features):
    """updates all features by contracting the given pair. In detail, it sets the features for the first node to
    the feature vector of the contracted pair and removes the features of the second node.

    Args:
        pair (:class:`ndarray`): pair that should be contracted
        features (:class:`ndarray`): features array that will be updated

    :rtype: :class:`ndarray`
    """
    cdef long v1
    v1 = <long>pair[1]

    if features.shape[0] > 0:
        features[v1] = pair[feature_offset:]

