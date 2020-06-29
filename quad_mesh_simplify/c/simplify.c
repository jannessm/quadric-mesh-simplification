#include "simplify.h"

#include <stdbool.h>

#include "mesh.h"
#include "q.h"
#include "edges.h"
#include "preserve_bounds.h"
#include "valid_pairs.h"
#include "targets.h"
#include "pair_heap.h"
#include "mesh_inversion.h"
#include "contract_pair.h"
#include "clean_mesh.h"

#define DEBUG
void _simplify_mesh(Mesh* mesh, unsigned int num_nodes, double threshold);

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

PyObject* simplify_mesh_c(PyObject* positions, PyObject* face, PyObject* features, unsigned int num_nodes, double threshold) {
  
  _import_array();
  int i, j;

  double* org_pos, * org_features;
  unsigned int* org_face;
  Mesh* mesh = malloc(sizeof(Mesh));
  
  mesh->n_vertices = PyArray_DIM((PyArrayObject*) positions, 0);
  mesh->n_face = PyArray_DIM((PyArrayObject*) face, 0);
  mesh->feature_length = PyArray_DIM((PyArrayObject*) features, 1);

  // copy data to mesh
  mesh->positions = malloc(sizeof(double) * 3 * mesh->n_vertices);
  mesh->face = malloc(sizeof(unsigned int) * 3 * mesh->n_face);
  mesh->features = malloc(sizeof(double) * mesh->feature_length * mesh->n_vertices);
  org_pos = (double*) PyArray_DATA((PyArrayObject*) positions);
  org_face = (unsigned int*) PyArray_DATA((PyArrayObject*) face);
  org_features = (double*) PyArray_DATA((PyArrayObject*) features);

  for (i = 0; i < mesh->n_vertices; i ++) {
    for (j = 0; j < 3; j++) {
      mesh->positions[i*3 + j] = org_pos[i * 3 + j];
    }
    for (j = 0; j < mesh->feature_length; j++) {
      mesh->features[i*mesh->feature_length + j] = org_features[i*mesh->feature_length + j];
    }
  }

  for (i = 0; i < mesh->n_face * 3; i++) {
    mesh->face[i] = org_face[i];
  }

  _simplify_mesh(mesh, num_nodes, threshold);

  npy_intp dim_pos[2] = {mesh->n_vertices, 3};
  npy_intp dim_face[2] = {mesh->n_face, 3};
  npy_intp dim_features[2] = {mesh->n_vertices, mesh->feature_length};

  PyObject* tuple = mesh->feature_length > 0 ? PyTuple_New(3) : PyTuple_New(2);

  PyObject* new_positions = PyArray_SimpleNewFromData(2, dim_pos, NPY_DOUBLE, (void*) mesh->positions);
  PyObject *capsule_pos = PyCapsule_New(mesh->positions, NULL, capsule_cleanup);
  PyArray_SetBaseObject((PyArrayObject *) new_positions, capsule_pos);
  PyTuple_SetItem(tuple, 0, new_positions);
  
  PyObject* new_face = PyArray_SimpleNewFromData(2, dim_face, NPY_UINT32, mesh->face);
  PyObject *capsule_face = PyCapsule_New(mesh->face, NULL, capsule_cleanup);
  PyArray_SetBaseObject((PyArrayObject *) new_face, capsule_face);
  PyTuple_SetItem(tuple, 1, new_face);
  
  PyObject* new_features = NULL;
  if (mesh->feature_length > 0) {
    new_features = PyArray_SimpleNewFromData(2, dim_features, NPY_DOUBLE, mesh->features);
    PyObject *capsule_features = PyCapsule_New(mesh->positions, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) new_features, capsule_features);
    PyTuple_SetItem(tuple, 2, new_features);
  }
  
  return tuple;
}

void _simplify_mesh(Mesh* mesh, unsigned int num_nodes, double threshold) {
  double* Q = compute_Q(mesh);

  SparseMat* edges = create_edges(mesh);

  preserve_bounds(mesh, Q, edges);

  Array2D_uint* valid_pairs = compute_valid_pairs(mesh, edges, threshold);

  PairList* targets = compute_targets(mesh, Q, valid_pairs);

  PairHeap* heap = list_to_heap(targets);

  bool* deleted_positions = calloc(mesh->n_vertices, sizeof(bool));
  bool* deleted_faces = calloc(mesh->n_face, sizeof(bool));

  Pair* p;
  unsigned int num_deleted_nodes = 0, i;

  while (mesh->n_vertices - num_deleted_nodes > num_nodes && heap->length > 0) {
    p = heap_pop(heap);

    if (p->v1 == p->v2 || deleted_positions[p->v1] || deleted_positions[p->v2]) {
      continue;
    }

    if (has_mesh_inversion(p->v1, p->v2, mesh, p->target, deleted_faces)) {
      continue;
    }

    for (i = 0; i < 3; i++) {
      mesh->positions[p->v1 * 3 + i] = p->target[i];
    }

    deleted_positions[p->v2] = true;

    // update Q
    for (i = 0; i < 16; i++) {
      Q[p->v1 * 16 + i] = Q[p->v1 * 16 + i] + Q[p->v2 * 16 + i];
    }

    update_face(mesh, deleted_faces, p->v1, p->v2);
    update_pairs(heap, mesh, Q, p->v1, p->v2);
    update_features(mesh, p);

    pair_free(p);
    num_deleted_nodes++;
  }

  clean_face(mesh, deleted_faces, deleted_positions);
  clean_positions_and_features(mesh, deleted_positions);

  sparse_free(edges);
  array_free(valid_pairs);
  heap_free(heap);
  free(deleted_positions);
  free(deleted_faces);
}
