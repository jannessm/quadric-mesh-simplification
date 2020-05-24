#include <stdio.h>
#include <stdbool.h>
#include <omp.h>
#include <stdlib.h>
#include "mesh.h"
#include "maths.h"

char get_edge(char* edges, unsigned int n, unsigned int v1, unsigned int v2) {
  unsigned int row = v1 < v2 ? v1 : v2;
  unsigned int column = row == v2 ? v1 : v2;
  return edges[row * n + column];
}

void set_edge(char* edges, unsigned int n, unsigned int v1, unsigned int v2, char value) {
  unsigned int row = v1 < v2 ? v1 : v2;
  unsigned int column = row == v2 ? v1 : v2;
  edges[row * n + column] = value;
}

void preserve_bounds(Mesh mesh, double* Q) {
  unsigned int i, j, k, a, v1, v2;

  unsigned int N = (mesh.n_vertices + 1) * mesh.n_vertices / 2;
  char *edges = malloc(N * sizeof(char));
  
  for (i = 0; i < mesh.n_vertices; i++) {
    for (j = i; j < mesh.n_vertices; j++) {
      set_edge(edges, mesh.n_vertices, i, j, 0);
    }
  }

  // create edges
  #pragma omp parallel for shared(edges, N) private(i, j, a, v1, v2)
  for (i = 0; i < mesh.n_face; i++) {
    for (j = 0; j < 3; j++) {
      a = (j + 1) % 3;
      v1 = mesh.face[i * 3 + j];
      v2 = mesh.face[i * 3 + a];

      // edge was not seen before
      if (get_edge(edges, mesh.n_vertices, v1, v2) == 0) {
        #pragma omp critical
        set_edge(edges, N, v1, v2, 1);
      } else if (get_edge(edges, mesh.n_vertices, v1, v2) == 1) {
        #pragma omp critical
        set_edge(edges, mesh.n_vertices, v1, v2, 2);
      }
    }
  }

  for (i = 0; i < mesh.n_vertices; i++) {
    for (j = i + 1; j < mesh.n_vertices; j++) {
      printf("%d, %d: %d\n", i, j, get_edge(edges, mesh.n_vertices, i, j));
    }
  }

  double *pos1, *pos2, *pos3, *p, *K;
  bool proc1, proc2;
  #ifdef _OPENMP
  omp_lock_t q_locks[mesh.n_vertices];

  for (i = 0; i < mesh.n_vertices; i++) {
    omp_init_lock(&(q_locks[i]));
  }
  #endif

  // add penalties
  #pragma omp parallel for private(i, j, k, a, v1, v2, pos1, pos2, pos3, p, K, proc1, proc2) shared(q_locks, Q)
  for (i = 0; i < mesh.n_face; i++) {
    for (j = 0; j < 3; j++) {
      a = (j + 1) % 3;
      v1 = mesh.face[i * 3 + j];
      v2 = mesh.face[i * 3 + a];

      // edge was seen once and therefore is a border of the mesh
      if (get_edge(edges, mesh.n_vertices, v1, v2) == 1) {
        pos1 = &mesh.positions[v1];
        pos2 = &mesh.positions[v2];
        pos3 = malloc(sizeof(double) * 3);

        pos3[0] = pos2[0] - pos1[0];
        pos3[1] = pos2[1] - pos1[1];
        pos3[2] = pos2[2] - pos1[2];

        p = normal(pos1, pos2, pos3);
        K = calculate_K(p);

        for (k = 0; k < 16; k++) {
          K[k] *= 1000;
        }
        
        proc1 = false;
        proc2 = false;

        //update Q
        #ifdef _OPENMP
        printf("%d %d for %d, %d\n", proc1, proc2, v1, v2);
        while (!proc1 || !proc2) {
          if (!proc1 && omp_test_lock(&(q_locks[v1]))) {
            add_K_to_Q(&(Q[v1]), K);
            omp_unset_lock(&(q_locks[v1]));
            proc1 = true;
          }
          
          if (!proc2 && omp_test_lock(&(q_locks[v2]))) {
            add_K_to_Q(&(Q[v2]), K);
            omp_unset_lock(&(q_locks[v2]));
            proc2 = true;
          }
        }
        #endif
        #ifndef _OPENMP
          add_K_to_Q(&(Q[v1]), K);
          add_K_to_Q(&(Q[v2]), K);
        #endif

        // free(p);
        // free(K);
        // free(pos3);
      }
    }
  }

  // free(edges);

  #ifdef _OPENMP
  for (i = 0; i < mesh.n_vertices; i++) {
    omp_destroy_lock(&(q_locks[i]));
  }
  #endif
}