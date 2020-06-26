#include <stdio.h>
#include <stdbool.h>
#include <omp.h>
#include <stdlib.h>
#include "mesh.h"
#include "maths.h"
#include "sparse_mat.h"


void preserve_bounds(Mesh* mesh, double* Q, SparseMat* edges) {
  unsigned int i, j, k, l, a, v1, v2, v3;

  double *pos1, *pos2, *pos3, *p, *K, *n;
  bool proc1, proc2;
  #ifdef _OPENMP
  omp_lock_t q_locks[mesh->n_vertices];

  for (i = 0; i < mesh->n_vertices; i++) {
    omp_init_lock(&(q_locks[i]));
  }
  #endif

  // add penalties
  #pragma omp parallel for private(i, j, k, l, a, v1, v2, v3, pos1, pos2, pos3, p, n, K, proc1, proc2) shared(q_locks, Q, mesh, edges)
  for (i = 0; i < mesh->n_face; i++) {
    for (j = 0; j < 3; j++) {
      a = (j + 1) % 3;
      v1 = mesh->face[i * 3 + j] < mesh->face[i * 3 + a] ? mesh->face[i * 3 + j] : mesh->face[i * 3 + a];
      v2 = mesh->face[i * 3 + j] == v1 ? mesh->face[i * 3 + a] : mesh->face[i * 3 + j];
      v3 = mesh->face[i * 3 + (j + 2) % 3];

      // edge was seen once and therefore is a border of the mesh
      if (sparse_get(edges, v1, v2) == 1) {
        pos1 = &(mesh->positions[v1*3]);
        pos2 = &(mesh->positions[v2*3]);
        pos3 = &(mesh->positions[v3*3]);

        printf("%u, %u, %u\n", v1, v2, v3);
        for(l = 0; l < 3; l++) {
          printf("%f %f %f\n", pos1[l], pos2[l], pos3[l]);
        }

        n = normal(pos1, pos2, pos3);
        p = normal(pos1, pos2, n);
        K = calculate_K(p);

        printf("%u, %u: ", v1, v2);
        for (k = 0; k < 16; k++) {
          printf("%f ", K[k]);
          K[k] *= 1000;
        }
        printf("\n");
        
        proc1 = false;
        proc2 = false;

        //update Q
        #ifdef _OPENMP
        while (!proc1 || !proc2) {
          if (!proc1 && omp_test_lock(&(q_locks[v1]))) {
            add_K_to_Q(&(Q[v1 * 16]), K);
            omp_unset_lock(&(q_locks[v1]));
            proc1 = true;
          }
          
          if (!proc2 && omp_test_lock(&(q_locks[v2]))) {
            add_K_to_Q(&(Q[v2 * 16]), K);
            omp_unset_lock(&(q_locks[v2]));
            proc2 = true;
          }
        }
        #endif
        #ifndef _OPENMP
          add_K_to_Q(&(Q[v1 * 16]), K);
          add_K_to_Q(&(Q[v2 * 16]), K);
        #endif

        free(n);
        free(p);
        free(K);
      }
    }
  }

  #ifdef _OPENMP
  for (i = 0; i < mesh->n_vertices; i++) {
    omp_destroy_lock(&(q_locks[i]));
  }
  #endif
}