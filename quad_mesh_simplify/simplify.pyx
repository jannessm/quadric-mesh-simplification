import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

from .preserve_bounds cimport preserve_bounds
from .q cimport compute_Q
from .targets cimport compute_targets
from .valid_pairs cimport compute_valid_pairs
from .contract_pair cimport update_pairs, update_face, update_features
from .mesh_inversion cimport has_mesh_inversion
from .maths cimport add_2D
from .heap cimport PairHeap

from testing_utils import plot_test_mesh

cimport cython
from cpython cimport array
import array

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def simplify_mesh(positions, face_in, num_nodes, features=None, threshold=0.):
    r"""simplify a mesh by contracting edges using the algortihm from `"Surface Simplification Using Quadric Error Metrics"
    <http://mgarland.org/files/papers/quadrics.pdf>`_.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        num_nodes (number): number of nodes that the final mesh will have
        threshold (number, optional): threshold of vertices distance to be a valid pair

    :rtype: (:class:`ndarray`, :class:`ndarray`)
    """
    cdef np.ndarray[DTYPE_LONG_T, ndim=2] face_, face__
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q_
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs, new_positions_, pos_, pos__, features_
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] pos1_, pos2_
    cdef double [:, :, :] Q
    cdef double [:, :] pos, new_positions
    cdef long [:, :] face, valid_pairs, face_view
    cdef double [:] pos1, pos2, p
    cdef int [:] sum_diminish
    cdef unsigned char [:] deleted_pos, deleted_faces
    cdef array.array deleted_pos_, deleted_faces_, sum_diminish_
    cdef long i, v1, v2, tmp
    cdef int update_failed, num_deleted_nodes, diminish_by
    num_deleted_nodes = 0
    tmp = 0

    assert(positions.shape[1] == 3)
    assert(face_in.shape[1] == 3)
    
    # copy positions, face and features for manipulation
    pos_ = np.copy(positions)
    new_positions_ = np.copy(positions)
    face_ = np.copy(face_in)
    pos = pos_
    new_positions = new_positions_
    face = face_

    deleted_pos_ = array.array('B', [])
    deleted_pos_ = array.clone(deleted_pos_, pos_.shape[0], True)
    deleted_pos = deleted_pos_

    deleted_face_ = array.array('B', [])
    deleted_face_ = array.clone(deleted_face_, face.shape[0], True)
    deleted_faces = deleted_face_

    pos1_ = np.zeros((3), dtype=DTYPE_DOUBLE)
    pos2_ = np.zeros((3), dtype=DTYPE_DOUBLE)
    pos1 = pos1_
    pos2 = pos2_
    
    # 1. compute Q for all vertices
    Q_ = compute_Q(pos, face)
    Q = Q_
    print('computed Q')
    # add penalty for boundaries
    preserve_bounds(pos, face, Q)
    print('preserved bounds')
    # 2. Select valid pairs
    valid_pairs = compute_valid_pairs(pos_, face_, threshold)
    print('computed valid pairs')

    # 3. compute optimal contration targets
    # of shape err, v1, v2, target, (features)    
    pairs = compute_targets(pos, Q, valid_pairs, features)
    print('computed targets')

    # 4. create heap sorted by costs
    cdef PairHeap heap = PairHeap(pairs)
    
    # 5. contract vertices until num_nodes reached
    while heap.length() > 0 and pos.shape[0] - num_deleted_nodes > num_nodes:
        p = heap.pop()
        v1 = <long>p[1]
        v2 = <long>p[2]

        # skip self-loops and already deleted nodes
        if v1 == v2 or deleted_pos[v1] or deleted_pos[v2]:
            continue

        #print(v1, v2, p[0])

        # store values for possible invalid contraction (inverted faces)
        pos1[...] = pos[v1]
        pos2[...] = pos[v2]
        # update positions if no mesh inversion is created
        new_positions[v1] = p[3:6]

        reverse_update = has_mesh_inversion(
            v1,
            v2,
            pos,
            new_positions,
            face,
            deleted_faces)

        if reverse_update:
            pos[v1, ...] = pos1
            continue
        else:
            for i in range(3):
                pos[v1, i] = p[3 + i]
            deleted_pos[v2] = True

        # if contraction is valid do updates
        add_2D(Q[v1], Q[v2], Q[v1])
        update_face(v1, v2, face, deleted_faces)
        update_features(p, features)
        update_pairs(v1, v2, heap, pos, Q, features)

        num_deleted_nodes += 1

        # delete positions, faces and features
        pos__ = pos_[[ not deleted for deleted in deleted_pos_.tolist()]]
        face__ = face_[[not deleted for deleted in deleted_face_.tolist()]]
        face_view = face__
        sum_diminish_ = array.array('i', [])
        sum_diminish_ = array.clone(sum_diminish_, deleted_pos.shape[0], True)
        sum_diminish = sum_diminish_
        diminish_by = 0
        
        for i in range(deleted_pos.shape[0]):
            sum_diminish[i] += diminish_by
            
            # if node was deleted diminish next nodes by 1
            if deleted_pos[i]:
                diminish_by += 1

        for i in range(face_view.shape[0]):
            for j in range(face_view.shape[1]):
                face_view[i, j] -= sum_diminish[face_view[i, j]]
        
        #plot_test_mesh(pos__, face__)

    #print(face_)
    if features is not None:
        features_ = np.copy(features)
        features_ = features_[[ not deleted for deleted in deleted_pos_.tolist()]]
        return pos__, face__, features_
    else:
        return pos__, face__
