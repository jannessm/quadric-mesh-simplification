import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.uint8_t DTYPE_UINT8_T

cdef extern from "math.h" nogil:
  double fabs(double x)

cimport cython
from cpython cimport array
import array
from .maths cimport normal, dot1d

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef int has_mesh_inversion(
    long v1,
    long v2,
    double [:, :] positions,
    double [:, :] new_positions,
    long [:, :] face,
    char [:] deleted_faces):
    """tests if a contraction of two nodes led to an inverted face by comparing all neighboring faces of v1 and v2.

    Args:
        v1 (:class:`long`): node id
        v2 (:class:`long`): node id
        positions (:class:`ndarray`): node positions before the contraction
        new_positions (:class:`ndarray`): node positions after the contraction
        face (:class:`ndarray`): face before the contraction

    :rtype: :class:`int` whether or not a mesh was inverted
    """
    cdef int i, j, check_face

    cdef array.array new_norm_, old_norm_
    cdef new_norm, old_norm

    new_norm_ = array.array('d', [0,0,0,0])
    old_norm_ = array.array('d', [0,0,0,0])
    new_norm = new_norm_
    old_norm = old_norm_

    for i in range(face.shape[0]):
        if deleted_faces[i]:
            continue
        
        check_face = False
        for j in range(3):
            if v1 == face[i, j] or v2 == face[i, j]:
                check_face = True
        
        if check_face and flipped(
            v1,
            v2,
            positions,
            new_positions,
            face[i],
            old_norm,
            new_norm):
            
            return True
                
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
    cdef double angle
    cdef int old_pos, reset, i, i1, i2, i3, j

    # check for each vertex if normal flipps
    for i in range(3):
        i1 = face[(0 + i) % 3]
        i2 = face[(1 + i) % 3]
        i3 = face[(2 + i) % 3]
        v1 = positions[i1]
        v2 = positions[i2]
        v3 = positions[i3]

        normal(v1, v2, v3, old_norm)
        if i1 == v2_id and i2 != v2_id and i3 != v2_id:
            i1 = v1_id
        elif i1 != v2_id and i2 == v2_id and i3 != v2_id:
            i2 = v1_id
        elif i1 != v2_id and i2 == v2_id and i3 == v2_id:
            i3 = v1_id
        elif i1 == v2_id or i2 == v2_id or i3 == v2_id:
            return False # face will be deleted anyways
        
        v1 = new_positions[i1]
        v2 = new_positions[i2]
        v3 = new_positions[i3]
        
        normal(v1, v2, v3, new_norm)

        old_norm[3] = 0 #normal includes d
        new_norm[3] = 0
        angle = dot1d(old_norm, new_norm)
        if angle < 0:
            return True

    return False
    