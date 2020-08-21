#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "test_utils.h"
#include "mesh.h"
#include "edges.h"
#include "pair.h"
#include "targets.h"

void test_compute_attributes() {
  const char* test_case = "targets compute attributes";
  unsigned int i;

  double positions[] = {
    -1., -1., -1.,
		-.5, 0., 0.,
		-1., 1., 1.,
		0., 0.25, 0.25,
		0., -0.25, -0.25,
		1., -1., -1.,
		.5, 0., 0.,
		1., 1., 1.,
		0., -1., -1.,
		0., 1., 1.
  };

  unsigned int face[] = {
    0, 1, 4,
    1, 3, 4,
    1, 2, 3,
    3, 6, 7,
    3, 4, 6,
    4, 5, 6,
    0, 8, 4,
    5, 4, 8,
    2, 3, 9,
    3, 9, 7
  };

  double q[] = {
    10000, -60000,  80000,  14000,
    -60000,  85000, -10000, -50000,
    80000, -10000,  15000,  53000,
    14000, -50000,  53000,  15000,

    0., 0., 0., 0.,
    0., 2.,-2., 0.,
    0.,-2., 2., 0.,
    0., 0., 0., 0.,

    10000, -60000,  80000,  14000,
    -60000,  85000, -10000, -50000,
    80000, -10000,  15000,  53000,
    14000, -50000,  53000,  15000,
  
    0., 0., 0., 0.,
    0., 3.,-3., 0.,
    0.,-3., 3., 0.,
    0., 0., 0., 0.,
    
    0., 0., 0., 0.,
    0., 3.,-3., 0.,
    0.,-3., 3., 0.,
    0., 0., 0., 0.,
    
    10000, -60000,  80000,  14000,
    -60000,  85000, -10000, -50000,
    80000, -10000,  15000,  53000,
    14000, -50000,  53000,  15000,
    
    0., 0., 0., 0.,
    0., 2.,-2., 0.,
    0.,-2., 2., 0.,
    0., 0., 0., 0.,
    
    10000, -60000,  80000,  14000,
    -60000,  85000, -10000, -50000,
    80000, -10000,  15000,  53000,
    14000, -50000,  53000,  15000,
    
    10000, -60000,  80000,  14000,
    -60000,  85000, -10000, -50000,
    80000, -10000,  15000,  53000,
    14000, -50000,  53000,  15000,
    
    10000, -60000,  80000,  14000,
    -60000,  85000, -10000, -50000,
    80000, -10000,  15000,  53000,
    14000, -50000,  53000,  15000,
  };

  Mesh m = {positions, NULL, face, 10, 10};

  Pair* pair = calculate_pair_attributes(&m, q, 1, 3);

  if (pair->error != 0) {
    fprintf(stderr, "  expected error %.4f got %.4f\n", 0.0, pair->error);
  }

  if (pair->v1 != 1) {
    fprintf(stderr, "  expected v1 %u got %u\n", 1, pair->v1);
  }

  if (pair->v2 != 3) {
    fprintf(stderr, "  expected v2 %u got %u\n", 3, pair->v2);
  }

  if (pair->target[0] != 0 || pair->target[1] != 0.25 || pair->target[2] != 0.25) {
    fprintf(stderr, "  expected target %.4f %.4f %.4f    got %.4f %.4f %.4f\n",
      pair->target[0], pair->target[1], pair->target[2],
      0.0, 0.25, 0.25
    );
  }

  pair_free(pair);

  printf("✓ %s\n", test_case);
}

void test_compute_targets() {
  const char* test_case = "targets compute targets";
  unsigned int i;

  double positions[] = {
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    1, 1, 1,
    1, 0, 1
  };

  unsigned int face[] = {
    0, 1, 2,
    1, 2, 3
  };

  unsigned int data[4] = {
    0, 1,
    1, 2
  };
  Array2D_uint* valid_pairs = malloc(sizeof(Array2D_uint));
  valid_pairs->rows = 2;
  valid_pairs->columns = 2;
  valid_pairs->data = data;

  double q[] = {
    2., 2., 2., 2.,
    2., 2., 2., 2.,
    2., 2., 2., 2.,
    2., 2., 2., 2.,

    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,

    2., 2., 2., 2.,
    2., 2., 2., 2.,
    2., 2., 2., 2.,
    2., 2., 2., 2.,

    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
  };

  Mesh m = {positions, NULL, face, 10, 10};

  PairList* pairs = compute_targets(&m, q, valid_pairs);

  double targets[] = {
    0, 0, 0,
    1, 0, 0
  };

  double err[] = {
    3, 12
  };

  for (i = 0; i < 2; i++) {
    if (pairs->list[i]->error != err[i]) {
      fprintf(stderr, "  expected error %.4f got %.4f\n", err[i], pairs->list[i]->error);
    }
    if (pairs->list[i]->v1 != valid_pairs->data[i * 2]) {
      fprintf(stderr, "  expected v1 %u got %u\n", valid_pairs->data[i * 2], pairs->list[i]->v1);
    }

    if (pairs->list[i]->v2 != valid_pairs->data[i * 2 + 1]) {
      fprintf(stderr, "  expected v2 %u got %u\n", valid_pairs->data[i * 2 + 1], pairs->list[i]->v2);
    }

    if (pairs->list[i]->target[0] != targets[i * 3] ||
        pairs->list[i]->target[1] != targets[i * 3 + 1] ||
        pairs->list[i]->target[2] != targets[i * 3 + 2]) {
      fprintf(stderr, "  expected target %.4f %.4f %.4f    got %.4f %.4f %.4f\n",
        pairs->list[i]->target[0], pairs->list[i]->target[1], pairs->list[i]->target[2],
        0.0, 0.25, 0.25
      );
    }
  }

  pairlist_free(pairs);
  free(valid_pairs);

  printf("✓ %s\n", test_case);
}

int main(void) {
  test_compute_attributes();
  test_compute_targets();
}