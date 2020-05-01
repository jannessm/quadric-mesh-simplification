import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.uint8_t DTYPE_UINT8_T

cimport cython
from cpython cimport array
import array
from .maths cimport normal, dot1d
from .utils cimport get_rows, face_normal

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef int has_mesh_inversion(
    long v1,
    long v2,
    double [:, :] positions,
    double [:, :] new_positions,
    long [:, :] face,
    unsigned char [:] deleted_faces):
    """tests if a contraction of two nodes led to an inverted face by comparing all neighboring faces of v1 and v2.

    Args:
        v1 (:class:`long`): node id
        v2 (:class:`long`): node id
        positions (:class:`ndarray`): node positions before the contraction
        new_positions (:class:`ndarray`): node positions after the contraction
        face (:class:`ndarray`): face before the contraction

    :rtype: :class:`int` whether or not a mesh was inverted
    """
    cdef int i, j

    cdef array.array new_norm_, old_norm_
    cdef new_norm, old_norm

    new_norm_ = array.array('d', [0,0,0,0])
    old_norm_ = array.array('d', [0,0,0,0])
    new_norm = new_norm_
    old_norm = old_norm_

    for i in range(face.shape[0]):
        if deleted_faces[i]:
            continue
        
        for j in range(3):
            if v1 == face[i, j] and flipped(
                    v1,
                    v2,
                    positions,
                    new_positions,
                    face[i],
                    old_norm,
                    new_norm):
                return True
            elif v2 == face[i, j]:
                if flipped(
                        v1,
                        v2,
                        positions,
                        new_positions,
                        face[i],
                        old_norm,
                        new_norm):
                    return True
                face[i, j] = v2
                
    return False

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int flipped(
    int v1_id,
    int v2_id,
    double [:, :] positions,
    double [:, :] new_positions,
    long [:] face,
    double [:] old_norm,
    double [:] new_norm):
    """calculates all normals for each face indexed by rows. The reference node prevents inverted normals.

    Args:
        positions (:class:`ndarray`): node positions
        face (:class:`ndarray`): face to calculate normals for
        rows (:class:`ndarray`): rows limitating the calculations for rows of interest
        reference_id (:class:`long`): node id of reference point

    :rtype: :class:`ndarray` normals
    """
    cdef double[:] v1, v2, v3
    cdef int old_pos, reset
    v1 = positions[face[0]]
    v2 = positions[face[1]]
    v3 = positions[face[2]]

    normal(v1, v2, v3, old_norm)

    reset = False
    for old_pos in range(3):
        if face[old_pos] == v2_id:
            face[old_pos] = v1_id
            reset = True
            break

    v1 = positions[face[0]]
    v2 = positions[face[1]]
    v3 = positions[face[2]]
    normal(v1, v2, v3, new_norm)

    if reset:
        face[old_pos] = v2_id

    old_norm[3] = 0
    new_norm[3] = 0

    return dot1d(old_norm, new_norm) < 0
    