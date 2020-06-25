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

PyObject* simplify_mesh_c(PyObject* positions, PyObject* face, PyObject* features, unsigned int num_nodes, double threshold) {
  
  Mesh* mesh = malloc(sizeof(Mesh));
  
  mesh->positions = (double*) PyArray_DATA((PyArrayObject*) positions);
  mesh->face = (unsigned int*) PyArray_DATA((PyArrayObject*) face);
  mesh->features = (double*) PyArray_DATA((PyArrayObject*) features);
  mesh->n_vertices = PyArray_DIM((PyArrayObject*) positions, 0);
  mesh->n_face = PyArray_DIM((PyArrayObject*) face, 0);
  mesh->feature_length = PyArray_DIM((PyArrayObject*) features, 1);

#ifdef DEBUG
  printf("\ngot mesh with n_vert %d n_face %d and features_len %d\n", mesh->n_vertices, mesh->n_face, mesh->feature_length);
  print_array_double(mesh->positions, mesh->n_vertices, 3);
  printf("\n");
  print_array_uint(mesh->face, mesh->n_face, 3);
#endif

  _simplify_mesh(mesh, num_nodes, threshold);

  npy_intp dim_pos[2], dim_face[2], dim_features[2];
  dim_pos[0] = dim_features[0] = mesh->n_vertices;
  dim_face[0] = mesh->n_face;
  dim_pos[1] = dim_face[1] = 3;
  dim_features[1] = mesh->feature_length;

  PyObject* new_positions = PyArray_SimpleNewFromData(2, dim_pos, NPY_DOUBLE, mesh->positions);
  PyObject* new_face = PyArray_SimpleNewFromData(2, dim_face, NPY_UINT, mesh->face);
  PyObject* new_features = PyArray_SimpleNewFromData(2, dim_features, NPY_DOUBLE, mesh->features);

  return PyTuple_Pack(3, new_positions, new_face, new_features);
}

void _simplify_mesh(Mesh* mesh, unsigned int num_nodes, double threshold) {
  double* Q = compute_Q(mesh);

  SparseMat* edges = create_edges(mesh);

#ifdef DEBUG
  preserve_bounds(mesh, Q, edges);
  printf("\n");
  print_Q(Q, 0, mesh->n_vertices);
#endif
  Array2D_uint* valid_pairs = compute_valid_pairs(mesh, edges, threshold);

#ifdef DEBUG
  printf("\n");
  print_array_uint(valid_pairs->data, valid_pairs->rows, valid_pairs->columns);
#endif

  PairList* targets = compute_targets(mesh, Q, valid_pairs);

  PairHeap* heap = list_to_heap(targets);

  bool* deleted_positions = calloc(mesh->n_vertices, sizeof(bool));
  bool* deleted_faces = calloc(mesh->n_face, sizeof(bool));

  Pair* p;
  unsigned int num_deleted_nodes = 0, i;

  while (mesh->n_vertices - num_deleted_nodes < num_nodes && heap->length > 0) {
    printf("so far so good\n");
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

    for (i = 0; i < 16; i++) {
      Q[p->v1 * 16 + i] = Q[p->v2 * 16 + i] + Q[p->v2 * 16 + i];
    }

    update_pairs(heap, mesh, Q, p->v1, p->v2);
    update_features(mesh, p);
    update_face(mesh, deleted_faces, p->v1, p->v2);

    pair_free(p);
  }

  clean_positions_and_features(mesh, deleted_positions);
  clean_face(mesh, deleted_faces, deleted_positions);

  sparse_free(edges);
  array_free(valid_pairs);
  heap_free(heap);
  free(deleted_positions);
  free(deleted_faces);
}
