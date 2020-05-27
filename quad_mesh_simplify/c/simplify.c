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

PyTupleObject simplify_mesh_c(PyArray_OBJECT* positions, PyArray_OBJECT* face, PyArray_OBJECT* features, unsigned int num_nodes, double threshold) {
  Mesh* mesh = malloc(sizeof(Mesh));
  
  mesh->positions = (double*) PyArray_DATA(positions);
  mesh->face = (unsigned int*) PyArray_DATA(face);
  mesh->features = (double*) PyArray_DATA(feature);
  mesh->n_vertices = PyArray_DIM(positions, 0);
  mesh->n_faces = PyArray_DIM(face, 0);
  mesh->feature_length = PyArray_DIM(features, 1);

  _simplify_mesh(mesh, num_nodes, threshold);
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

  while (mesh->n_vertices - num_deleted_nodes < num_nodes && heap->length > 0) {
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