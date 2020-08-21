#include <stdlib.h>
#include <stdbool.h>
#include "mesh.h"

void clean_positions_and_features(Mesh* mesh, bool* deleted_pos);

void clean_face(Mesh* mesh, bool* deleted_faces, bool* deleted_positions);