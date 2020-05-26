#include "array.h"
#include "mesh.h"
#include "sparse_mat.h"

Array2D_uint* compute_valid_pairs(Mesh* mesh, SparseMat* edges, unsigned int threshold);