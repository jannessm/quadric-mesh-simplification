import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

from preserve_bounds cimport preserve_bounds
from q cimport compute_Q
from targets cimport compute_targets
from valid_pairs cimport compute_valid_pairs
from contract_pair cimport update_positions, update_Q, update_pairs, update_face, update_features, sort_by_error
from mesh_inversion cimport has_mesh_inversion

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
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] pos1, pos2
    cdef long i
    cdef int update_failed

    # 1. compute Q for all vertices
    Q = compute_Q(positions, face)

    # add penalty for boundaries
    preserve_bounds(positions, face, Q)

    # 2. Select valid pairs
    valid_pairs = compute_valid_pairs(positions, face, threshold)

    # 3. compute optimal contration targets
    # of shape err, v1, v2, target, (features)    
    pairs = compute_targets(positions, Q, valid_pairs, features)

    # 4. create head sorted by costs
    pairs = sort_by_error(pairs)

    # 5. contract vertices until num_nodes reached
    while ( positions.shape[0] > num_nodes and
            pairs.shape[0] > 0):

        # skip self-loops
        if pairs[0, 1] == pairs[0, 2]:
            pairs = np.delete(pairs, 0, 0)
            continue


        # store values for possible invalid contraction (inverted faces)
        pos1 = positions[<long>pairs[0, 1]]
        pos2 = positions[<long>pairs[0, 2]]

        # update positions if no mesh inversion is created
        new_positions = update_positions(pairs[0], positions)

        reverse_update = has_mesh_inversion(
            <long>pairs[0, 1],
            <long>pairs[0, 2],
            positions,
            new_positions,
            face)

        if reverse_update:
            positions[pairs[0, 1]] = pos1
            positions = np.insert(positions, pairs[0, 2], pos2, axis=0)
            continue
        else:
            positions = new_positions

        # if contraction is valid do updates
        Q = update_Q(pairs[0], Q)
        face = update_face(pairs[0], face)
        features = update_features(pairs[0], features)
        pairs = update_pairs(pairs, positions, Q, features)

    if features is not None:
        return positions, face, features
    else:
        return positions, face
