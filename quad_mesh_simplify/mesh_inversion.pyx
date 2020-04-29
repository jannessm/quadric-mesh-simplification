import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.uint8_t DTYPE_UINT8_T

cimport cython

from .utils cimport get_rows, face_normal

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef int has_mesh_inversion(
    long v1,
    long v2,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] new_positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face):
    """tests if a contraction of two nodes led to an inverted face by comparing all neighboring faces of v1 and v2.

    Args:
        v1 (:class:`long`): node id
        v2 (:class:`long`): node id
        positions (:class:`ndarray`): node positions before the contraction
        new_positions (:class:`ndarray`): node positions after the contraction
        face (:class:`ndarray`): face before the contraction

    :rtype: :class:`int` whether or not a mesh was inverted
    """
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] normals, new_normals, angles
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] rows, rows1, rows2
    cdef np.ndarray[DTYPE_UINT8_T, ndim=2] v2s
    
    # calculate face normals for each face with node v1
    rows1 = get_rows(face == v1)
    normals = calculate_face_normals(positions, face, rows1, v1)
    
    # calculate face normals for each face with ONLY node v2
    rows2 = get_rows(face == v2)
    rows2 = rows2[rows2 != v1]
    normals = np.vstack([
        normals,
        calculate_face_normals(positions, face, rows2, v2)
    ])
    
    # merge rows
    rows = np.append(rows1, rows2)
    
    # update face
    v2s = face == v2
    face[v2s] = v1
    face[face > v2] -= 1
    
    # calculate normals with updated positions
    new_normals = calculate_face_normals(new_positions, face, rows, v1)
    
    # revert face update
    face[face >= v2] += 1
    face[v2s] = v2

    # calculate angles between old and new normals
    angles = normals.dot(new_normals.T) * np.eye(rows.shape[0]) 
    
    # return True if at least one angle is greater than 90°
    return (angles < 0).sum() > 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] calculate_face_normals(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    np.ndarray[DTYPE_LONG_T, ndim=1] rows,
    long reference_id):
    """calculates all normals for each face indexed by rows. The reference node prevents inverted normals.

    Args:
        positions (:class:`ndarray`): node positions
        face (:class:`ndarray`): face to calculate normals for
        rows (:class:`ndarray`): rows limitating the calculations for rows of interest
        reference_id (:class:`long`): node id of reference point

    :rtype: :class:`ndarray` normals
    """
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] normals = np.zeros((face.shape[0], 3), dtype=DTYPE_DOUBLE)
    cdef long i, ref

    for i in rows:
        ref = np.where(face[i] != reference_id)[0][0]
        normals[i] = face_normal(positions[face[i]], True, ref)

    return normals[rows]