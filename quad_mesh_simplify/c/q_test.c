#include <stdbool.h>
#include "q.h"
#include "utils.h"
#include "mesh.h"

int main(void) {

  double positions[] = {
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    1, 1, 1
  };

  unsigned int face[] = {
    0, 1, 2,
    0, 2, 3
  };

  double res[] = {
    0.5, -0.5, 0.,  0. ,
   -0.5, 0.5, 0.,  0. ,
    0.,  0.,  1.,  0. ,
    0.,  0.,  0.,  0. ,

    0.,  0.,  0.,  0. ,
    0.,  0.,  0.,  0. ,
    0.,  0.,  1.,  0. ,
    0.,  0.,  0.,  0. ,

    0.5, -0.5, 0.,  0. ,
   -0.5, 0.5, 0.,  0. ,
    0.,  0.,  1.,  0. ,
    0.,  0.,  0.,  0. ,

    0.5, -0.5, 0.,  0. ,
   -0.5, 0.5, 0.,  0. ,
    0.,  0.,  0.,  0. ,
    0.,  0.,  0.,  0. ,
  };

  Mesh m = {positions, NULL, face, 4, 2};

  double* q = compute_Q(m);

  int i;
  for (i = 0; i < 16; i++) {
    if (res[i] - q[i] > 10e-6) {
      fprintf(stderr, "FAIL q_test:\n\nQ:");
      print_Q(q, 4, true);
      fprintf(stderr, "\nExpected:\n");
      print_Q(res, 4, true);
      exit(-2);
    }
  }

  printf("Test Q: done\n");
}