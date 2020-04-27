import numpy as np
cimport numpy as np

cpdef np.ndarray get_faces_for_node(long node_id, np.ndarray face):
	cdef np.ndarray rows = get_rows(face == node_id)
	return face[rows]

cpdef np.ndarray get_rows(np.ndarray condition):
	return np.unique(np.where(condition)[0])