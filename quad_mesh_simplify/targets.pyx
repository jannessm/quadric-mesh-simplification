import numpy as np

DTYPE_DOUBLE = np.double

from . cimport maths

from cpython cimport array
import array
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] compute_targets(
    double [:, :] positions,
    double [:, :, :] Q,
    long [:, :] valid_pairs,
    double [:, :] features):
    """Computes the optimal position and the resulting feature vector (if provided) of the contracted node for nodes in valid_pairs.
    Since the feature vector should be aggregated, the exact mixture of v1 and v2 are needed and sampled from 10 different positions
    on the edge v1, v2.

    Args:
        positions (:class:`ndarray`): array of shape (num_nodes, 3) containing the x, y, z position for each node
        Q (:class:`ndarray`): Q matrixes for each node (shape (num_nodes, 4, 4))
        valid_pairs (:class:`ndarray`): list of valid_pairs. A pair consists of the error, v1, v2, target position(, feature vector).
        features (:class:`ndarray`, optional): feature matrix of shape (num_nodes, num_features)

    :rtype: :class:`ndarray`
    """
    if features is None:
        features = np.zeros((0,0))
    
    # a pair consists of error, v1, v2, target, feature
    
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs
    cdef double [:, :] pairs_view
    cdef double [:] p
    cdef int pair_shape, target_offset, i, j

    pair_shape = 3 + 3 + features.shape[1]
    target_offset = 3

    pairs = np.zeros((valid_pairs.shape[0], pair_shape), dtype=DTYPE_DOUBLE)
    pairs_view = pairs

    for i in range(valid_pairs.shape[0]):
        calculate_pair_attributes(
            valid_pairs[i, 0],
            valid_pairs[i, 1],
            positions,
            Q,
            features,
            pairs_view[i, :])
        print(pairs_view[i])

    return pairs


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void calculate_pair_attributes(
    long v1,
    long v2,
    double [:, :] positions,
    double [:, :, :] Q,
    double [:, :] features,
    double [:] pair):
    """Computes the optimal position of the contracted node of the edge (v1, v2).

    Args:
        v1 (:class:`long`): id for first node
        v2 (:class:`long`): id for second node
        positions (:class:`ndarray`): array of shape (num_nodes, 3) containing the x, y, z position for each node
        Q (:class:`ndarray`): Q matrixes for each node (shape (num_nodes, 4, 4))
        features (:class:`ndarray`, optional): feature matrix of shape (num_nodes, num_features)

    :rtype: :class:`ndarray`
    """
    cdef int min_id, i, features_len

    if features is None:
        features_len = 0
    else:
        features_len = features.shape[1]

    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] new_Q
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] p1, p2, p12

    cdef double [:, :] new_Q_view
    cdef double [:] p1_view, p2_view, p12_view

    cdef int pair_shape = 3 + 3 + features_len
    cdef int target_offset = 3
    cdef int feature_offset = 3 + 3
    cdef double min_error, error

    new_Q = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
    new_Q_view = new_Q

    p1 = np.zeros((3), dtype=DTYPE_DOUBLE)
    p2 = np.zeros((3), dtype=DTYPE_DOUBLE)
    p12 = np.zeros((3), dtype=DTYPE_DOUBLE)
    p1_view = p1
    p2_view = p2
    p12_view = p12

    pair[1] = v1
    pair[2] = v2

    maths.add_2D(Q[v1], Q[v2], new_Q)
    
    # do not use explicit solution because of feature trade-off

    # calculate errors for a 10 different targets on p1 -> p2
    min_id = 0
    min_error = 10e10
    for i in range(11):
        p1_view[:] = positions[v1]
        p2_view[:] = positions[v2]
        maths.mul_scalar_1D(p1_view, i * 0.1)
        maths.mul_scalar_1D(p2_view, 1 - i * 0.1)
        for j in range(3):
            p12_view[j] = p1_view[j] + p2_view[j]
    
        error = maths.error(p12_view, new_Q_view)

        if error < min_error:
            min_error = error
            min_id = i
            pair[target_offset: feature_offset] = p12_view
            pair[0] = error

    if features_len != 0:
        for i in range(features_len):
            pair[feature_offset + i] =  min_id * 0.1 * features[v1, i] +  \
                                        (1 - min_id * 0.1) * features[v2, i]
