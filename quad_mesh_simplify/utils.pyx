import numpy as np
cimport numpy as np

cdef get_faces_for_node(long node_id, face):
	cdef int rows = np.unique(np.where(face == node_id)[0])
	return face[rows]