#include "pair.h"
#include "mesh.h"
#include "array.h"

PairList* compute_targets(Mesh* mesh, double* Q, Array2D_uint* valid_pairs);

Pair* calculate_pair_attributes(Mesh* mesh, double* Q, unsigned int v1, unsigned int v2);