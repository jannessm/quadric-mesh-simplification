#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "q.h"
#include "test_utils.h"
#include "mesh.h"

const char *test_case = "Q";

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

  double expected[] = {
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

  Mesh m = {positions, NULL, face, 4, 2, 0};

  double* q = compute_Q(&m);

  q_equal(test_case, expected, q, 0, 4 * 16);

  printf("✓ %s\n", test_case);

  free(q);
}