#include "mesh.h"
#include "maths.h"
#include "utils.h"
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

double* compute_Q(Mesh mesh) {
  #ifdef _OPENMP
  omp_lock_t q_locks[mesh.n_vertices];
  #endif
  double *K, *p, *Q;
  double *pos1, *pos2, *pos3;
  bool proc1, proc2, proc3;
  unsigned int i;

  Q = calloc(mesh.n_vertices * 4 * 4, sizeof(double));
  
  #ifdef _OPENMP
  for (i = 0; i < mesh.n_vertices; i++) {
    omp_init_lock(&(q_locks[i]));
  }
  #endif

  #pragma omp parallel for shared(q_locks, Q) private(K, p, pos1, pos2, pos3, proc1, proc2, proc3)
  for (i = 0; i < mesh.n_face; i++) {
    proc1 = false;
    proc2 = false;
    proc3 = false;

    pos1 = &mesh.positions[mesh.face[i*3] * 3];
    pos2 = &mesh.positions[mesh.face[i*3 + 1] * 3];
    pos3 = &mesh.positions[mesh.face[i*3 + 2] * 3];

    p = normal(pos1, pos2, pos3);
    K = calculate_K(p);

    //update Q
    #ifdef _OPENMP
    while (!proc1 || !proc2 || !proc3) {
      if (!proc1 && omp_test_lock(&(q_locks[mesh.face[i * 3]]))) {
        add_K_to_Q(&(Q[mesh.face[i * 3] * 16]), K);
        omp_unset_lock(&(q_locks[mesh.face[i * 3]]));
        proc1 = true;
      }
      
      if (!proc2 && omp_test_lock(&(q_locks[mesh.face[i * 3 + 1]]))) {
        add_K_to_Q(&(Q[mesh.face[i * 3 + 1] * 16]), K);
        omp_unset_lock(&(q_locks[mesh.face[i * 3 + 1]]));
        proc2 = true;
      }

      if (!proc3 && omp_test_lock(&(q_locks[mesh.face[i * 3 + 2]]))) {
        add_K_to_Q(&(Q[mesh.face[i * 3 + 2] * 16]), K);
        omp_unset_lock(&(q_locks[mesh.face[i * 3 + 2]]));
        proc3 = true;
      }
    }
    #endif
    #ifndef _OPENMP
      add_K_to_Q(&(Q[mesh.face[i * 3] * 16]), K);
      add_K_to_Q(&(Q[mesh.face[i * 3 + 1] * 16]), K);
      add_K_to_Q(&(Q[mesh.face[i * 3 + 2] * 16]), K);
    #endif
    
    free(K);
    free(p);
  }

  #ifdef _OPENMP
  for (i = 0; i < mesh.n_vertices; i++) {
    omp_destroy_lock(&(q_locks[i]));
  }
  #endif
  
  return Q;
}