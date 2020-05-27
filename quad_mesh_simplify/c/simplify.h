#include <numpy/arrayobject.h>
#include <stdbool.h>

void simplify_mesh_c(PyArray_OBJECT* positions, PyArray_OBJECT* face, PyArray_OBJECT* features, unsigned int num_nodes, double threshold);