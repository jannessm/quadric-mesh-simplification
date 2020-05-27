#include <stdbool.h>
#include "mesh.h"

bool has_mesh_inversion(unsigned int v1, unsigned int v2, Mesh* mesh, double* new_position, bool* deleted_faces);