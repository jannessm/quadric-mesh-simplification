from pair cimport Pair
cimport numpy as np

cpdef (np.ndarray, np.ndarray, np.ndarray) contract_first_pair(
		list,
		np.ndarray,
		np.ndarray)