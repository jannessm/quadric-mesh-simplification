import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array
import array

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

DEBUG = True

if DEBUG:
    from time import time

from .clean_mesh cimport clean_positions, clean_face, clean_features
from .contract_pair cimport update_pairs, update_face, update_features
from .heap cimport PairHeap
from .maths cimport add_2D
from .mesh_inversion cimport has_mesh_inversion
from .preserve_bounds cimport preserve_bounds
from .q cimport compute_Q
from .targets cimport compute_targets
from .valid_pairs cimport compute_valid_pairs


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

    if DEBUG:
        start = time()

    cdef np.ndarray[DTYPE_LONG_T, ndim=2] face_copy
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q_
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] pos_copy, features_copy, new_positions_, pairs
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] pos1_, pos2_
    
    cdef array.array deleted_pos_, deleted_faces_

    cdef PairHeap heap
    
    cdef double [:, :, :] Q
    cdef double [:, :] pos, new_positions, feats
    cdef double [:] pos1, pos2, p
    
    cdef long [:, :] face, valid_pairs, face_view
    cdef long i, v1, v2
    
    cdef char [:] deleted_pos, deleted_faces
    cdef char reverse_update
    
    cdef int update_failed, num_deleted_nodes, diminish_by

    assert(positions.shape[1] == 3)
    assert(face_in.shape[1] == 3)
    assert(num_nodes < positions.shape[0])
    
    num_deleted_nodes = 0
    tmp = 0

    # copy positions, face and features for manipulation
    pos_copy = np.copy(positions)
    pos = pos_copy
    
    new_positions_ = np.copy(positions)
    new_positions = new_positions_
    
    face_copy = np.copy(face_in)
    face = face_copy

    deleted_pos_ = array.array('B', [])
    deleted_pos_ = array.clone(deleted_pos_, pos.shape[0], True)
    deleted_pos = deleted_pos_

    deleted_face_ = array.array('B', [])
    deleted_face_ = array.clone(deleted_face_, face.shape[0], True)
    deleted_faces = deleted_face_

    if features is not None:
        features_copy = np.copy(features)
        feats = features_copy
    else:
        feats = None

    pos1_ = np.zeros((3), dtype=DTYPE_DOUBLE)
    pos2_ = np.zeros((3), dtype=DTYPE_DOUBLE)
    pos1 = pos1_
    pos2 = pos2_
    
    # 1. compute Q for all vertices
    Q_ = compute_Q(pos, face)
    Q = Q_

    # add penalty for boundaries
    preserve_bounds(pos, face, Q)

    # 2. Select valid pairs
    valid_pairs = compute_valid_pairs(pos, face, threshold)

    # 3. compute optimal contration targets
    # of shape err, v1, v2, target, (features)    
    pairs = compute_targets(pos, Q, valid_pairs, feats)

    # 4. create heap sorted by costs
    heap = PairHeap(pairs)
    
    if DEBUG:
        print('setup in {} sec'.format(time() - start))
        start = time()

    # 5. contract vertices until num_nodes reached
    while heap.length() > 0 and pos.shape[0] - num_deleted_nodes > num_nodes:
        p = heap.pop()
        v1 = <long>p[1]
        v2 = <long>p[2]

        # skip self-loops and already deleted nodes
        if v1 == v2 or deleted_pos[v1] or deleted_pos[v2]:
            continue

        # store values for possible invalid contraction (inverted faces)
        pos1[...] = pos[v1]
        pos2[...] = pos[v2]
        
        new_positions[v1] = p[3:6]

        # update positions if no mesh inversion is created
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
        update_features(p, feats)
        update_pairs(v1, v2, heap, pos, Q, features)

        num_deleted_nodes += 1

    # delete positions, faces and features
    pos_copy = clean_positions(pos_copy, deleted_pos_)
    face_copy = clean_face(face_copy, deleted_face_, deleted_pos)

    if DEBUG:
        print('reduction in {} sec'.format(time() - start))

    if features is not None:
        features_copy = clean_features(features_copy, deleted_pos_)
        return pos_copy, face_copy, features_copy
    else:
        return pos_copy, face_copy
