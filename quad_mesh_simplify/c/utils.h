#include <stdio.h>
#include <stdlib.h>

#include "mesh.h"

Mesh read_ply(char* filename);

void print_Q(double* q, unsigned int n_vertices, char to_stderr);