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

// #define DEBUG
#include <time.h>

void _simplify_mesh(Mesh* mesh, unsigned int num_nodes, double threshold, double max_err);

PyObject* simplify_mesh_c(PyObject* positions, PyObject* face, PyObject* features, unsigned int num_nodes, double threshold, double max_err) {
  
  _import_array();
  unsigned int i, j;
  Mesh* mesh = malloc(sizeof(Mesh));
  
  mesh->n_vertices = PyArray_DIM((PyArrayObject*) positions, 0);
  mesh->n_face = PyArray_DIM((PyArrayObject*) face, 0);
  mesh->feature_length = PyArray_DIM((PyArrayObject*) features, 1);

  // copy data to mesh
  mesh->positions = malloc(sizeof(double) * 3 * mesh->n_vertices);
  mesh->face = malloc(sizeof(unsigned int) * 3 * mesh->n_face);
  mesh->features = malloc(sizeof(double) * mesh->feature_length * mesh->n_vertices);

  for (i = 0; i < mesh->n_vertices; i ++) {
    for (j = 0; j < 3; j++) {
      mesh->positions[i * 3 + j] = *((double*) PyArray_GETPTR2((PyArrayObject*) positions, (npy_intp) i, (npy_intp) j));
    }
    for (j = 0; j < mesh->feature_length; j++) {
      mesh->features[i*mesh->feature_length + j] = *((double*) PyArray_GETPTR2((PyArrayObject*) features, (npy_intp) i, (npy_intp) j));
    }
  }

  for (i = 0; i < mesh->n_face; i++) {
    for (j = 0; j < 3; j++) {
      mesh->face[i * 3 + j] = *((unsigned int*) PyArray_GETPTR2((PyArrayObject*) face, i, j));
    }
  }

  _simplify_mesh(mesh, num_nodes, threshold, max_err);

  npy_intp dim_pos[2] = {mesh->n_vertices, 3};
  npy_intp dim_face[2] = {mesh->n_face, 3};
  npy_intp dim_features[2] = {mesh->n_vertices, mesh->feature_length};

  PyObject* tuple = mesh->feature_length > 0 ? PyTuple_New(3) : PyTuple_New(2);

  PyObject* new_positions = PyArray_SimpleNewFromData(2, dim_pos, NPY_DOUBLE, mesh->positions);
  PyTuple_SetItem(tuple, 0, new_positions);
  
  PyObject* new_face = PyArray_SimpleNewFromData(2, dim_face, NPY_UINT32, mesh->face);
  PyTuple_SetItem(tuple, 1, new_face);
  
  PyObject* new_features = NULL;
  if (mesh->feature_length > 0) {
    new_features = PyArray_SimpleNewFromData(2, dim_features, NPY_DOUBLE, mesh->features);
    PyTuple_SetItem(tuple, 2, new_features);
  }
  
  return tuple;
}

void _simplify_mesh(Mesh* mesh, unsigned int num_nodes, double threshold, double max_err) {

#ifdef DEBUG
  clock_t start = clock();
#endif

  double* Q = compute_Q(mesh);

  UpperTriangleMat* edges = create_edges(mesh);

  preserve_bounds(mesh, Q, edges);

  Array2D_uint* valid_pairs = compute_valid_pairs(mesh, edges, threshold);
  upper_free(edges);

  PairList* targets = compute_targets(mesh, Q, valid_pairs);
  array_free(valid_pairs);

  PairHeap* heap = list_to_heap(targets);

#ifdef DEBUG
  clock_t end = clock();
  float seconds = (float)(end - start) / CLOCKS_PER_SEC;
  printf("setup in %f seconds\n", seconds);
  start = clock();
#endif

  bool* deleted_positions = calloc(mesh->n_vertices, sizeof(bool));
  bool* deleted_faces = calloc(mesh->n_face, sizeof(bool));

  Pair* p;
  unsigned int num_deleted_nodes = 0, i;

  while (mesh->n_vertices - num_deleted_nodes > num_nodes && heap->length > 0) {
    // check for keyboard interrupt
    if (((mesh->n_vertices - num_deleted_nodes) % 250) == 0) {
      if(PyErr_CheckSignals() != 0) exit(-1);
    }
    
    p = heap_pop(heap);

    if (p->v1 == p->v2 || deleted_positions[p->v1] || deleted_positions[p->v2]) {
      continue;
    }

    if (p->error > max_err) {
      break;
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

  heap_free(heap);
  free(deleted_positions);
  free(deleted_faces);

  #ifdef DEBUG
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("reduction in %f seconds\n", seconds);
  #endif
}
