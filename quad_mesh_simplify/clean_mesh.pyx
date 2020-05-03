import array
import numpy as np

cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] clean_positions(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] pos,
    array.array deleted_pos):

    return pos[[ not deleted for deleted in deleted_pos.tolist()]]

cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] clean_features(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features,
    array.array deleted_pos):

    return features[[ not deleted for deleted in deleted_pos.tolist()]]

cdef np.ndarray[DTYPE_LONG_T, ndim=2] clean_face(
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    array.array deleted_faces,
    char [:] deleted_pos):

    cdef np.ndarray[DTYPE_LONG_T, ndim=2] new_face_
    cdef long [:, :] new_face

    cdef array.array sum_diminish_
    cdef int [:] sum_diminish
    cdef int diminish_by

    new_face_ = face[[not deleted for deleted in deleted_faces.tolist()]]
    new_face = new_face_

    sum_diminish_ = array.array('i', [])
    sum_diminish_ = array.clone(sum_diminish_, deleted_pos.shape[0], True)
    sum_diminish = sum_diminish_
    diminish_by = 0
    
    for i in range(deleted_pos.shape[0]):
        sum_diminish[i] += diminish_by
        
        # if node was deleted diminish next nodes by 1
        if deleted_pos[i]:
            diminish_by += 1

    for i in range(new_face.shape[0]):
        for j in range(new_face.shape[1]):
            new_face[i, j] -= sum_diminish[new_face[i, j]]

    return new_face_