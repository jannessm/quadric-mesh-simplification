import numpy as np
cimport numpy as np

cpdef np.ndarray get_faces_for_node(long node_id, np.ndarray face):
	rows = np.unique(np.where(face == node_id)[0])
	return face[rows]