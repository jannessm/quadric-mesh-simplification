#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

PyObject* simplify_mesh_c(PyObject* positions, PyObject* face, PyObject* features, unsigned int num_nodes, double threshold, double max_err);