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

from .heap cimport PairHeap

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def simplify_mesh(positions, face, num_nodes, features=None, threshold=0.):
    r"""simplify a mesh by contracting edges using the algortihm from `"Surface Simplification Using Quadric Error Metrics"
    <http://mgarland.org/files/papers/quadrics.pdf>`_.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        num_nodes (number): number of nodes that the final mesh will have
        threshold (number, optional): threshold of vertices distance to be a valid pair

    :rtype: (:class:`ndarray`, :class:`ndarray`)
    """
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q
    cdef np.ndarray[DTYPE_LONG_T, ndim=2] valid_pairs
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs, new_positions
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] pos1, pos2, p
    cdef list deleted_pos, deleted_faces
    cdef long i, v1, v2
    cdef int update_failed

    deleted_pos = []
    deleted_faces = []

    assert(positions.shape[1] == 3)
    assert(face.shape[1] == 3)
    
    # 1. compute Q for all vertices
    Q = compute_Q(positions, face)
    # add penalty for boundaries
    preserve_bounds(positions, face, Q)

    # 2. Select valid pairs
    valid_pairs = compute_valid_pairs(positions, face, threshold)

    # 3. compute optimal contration targets
    # of shape err, v1, v2, target, (features)    
    pairs = compute_targets(positions, Q, valid_pairs, features)

    # 4. create heap sorted by costs
    cdef PairHeap heap = PairHeap(pairs)

    new_positions = np.copy(positions)

    # 5. contract vertices until num_nodes reached
    while heap.length() > 0 and positions.shape[0] - deleted_pos.length > num_nodes:
        p = heap.pop()
        v1 = <long>p[1]
        v2 = <long>p[2]

        # skip self-loops
        if v1 == v2:
            continue

        # store values for possible invalid contraction (inverted faces)
        pos1, pos2 = np.copy(positions[[v1, v2]])

        # update positions if no mesh inversion is created
        new_positions[v1] = p[3:6]

        reverse_update = has_mesh_inversion(
            v1,
            v2,
            positions,
            new_positions,
            face)

        if reverse_update:
            positions[v1] = pos1
            continue
        else:
            positions[v1] = pos1
            deleted_pos.append(v2)

        # if contraction is valid do updates
        Q[v1] = Q[v1] + Q[v2]
        update_face(p, face, deleted_faces)
        update_features(p, features)
        update_pairs(v1, v2, heap, positions, Q, features)

    if features is not None:
        return positions, face, features
    else:
        return positions, face
