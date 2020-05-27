#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "contract_pair.h"
#include "test_utils.h"
#include "mesh.h"
#include "pair_heap.h"
#include "array.h"

void test_update_pairs() {
  const char* test_case = "update pairs";
  unsigned int i;

  double positions[] = {
    0., 0., 0.,
    1., 1., 1.,
    -1., 0., 1.,
    0., 1., -1
  };

  double q[] = {
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.,
    1., 1., 1., 1.
  };

  Mesh m = {positions, NULL, NULL, 5, 0, 0};

  Pair* p1 = pair_init(0);
  p1->error = 1;
  p1->v1 = 0;
  p1->v2 = 1;
  p1->target[0] = 9;
  p1->target[1] = 9;
  p1->target[2] = 9;


  Pair* p2 = pair_init(0);
  p2->error = 1;
  p2->v1 = 0;
  p2->v2 = 1;
  p2->target[0] = 8;
  p2->target[1] = 9;
  p2->target[2] = 9;


  Pair* p3 = pair_init(0);
  p3->error = 5;
  p3->v1 = 1;
  p3->v2 = 2;
  p3->target[0] = 9;
  p3->target[1] = 8;
  p3->target[2] = 9;

  Pair* p4 = pair_init(0);
  p4->error = -4;
  p4->v1 = 2;
  p4->v2 = 3;
  p4->target[0] = 9;
  p4->target[1] = 9;
  p4->target[2] = 8;

  Pair* p5 = pair_init(0);
  p5->error = 3;
  p5->v1 = 2;
  p5->v2 = 3;
  p5->target[0] = 9;
  p5->target[1] = 9;
  p5->target[2] = 8;

  PairList* list = pairlist_init();
  pairlist_append(list, p1);
  pairlist_append(list, p2);
  pairlist_append(list, p3);
  pairlist_append(list, p4);
  pairlist_append(list, p5);

  PairHeap* heap = list_to_heap(list);

  update_pairs(heap, &m, q, 0, 1);

  if (heap_get_error(heap, 1) != -10e6 || heap_get_error(heap, 2) != -10e6) {
    fprintf(stderr, "✗ %s:\n  error wasn't updated correctly. expected -10e6 got %.f and %.f\n",
      test_case, heap_get_error(heap, 1), heap_get_error(heap, 2));
    print_heap(heap);
    exit(-2);
  }

  Pair* pair;
  bool updated_correctly = false;
  for (i = 1; i < heap->length; i++) {
    pair = heap_get_pair(heap, i);
    if (
      pair->error == 2 &&
      pair->v1 == 0 &&
      pair->v2 == 2 &&
      pair->target[0] == -1 &&
      pair->target[1] == 0 &&
      pair->target[2] == 1
    ) {
      updated_correctly = true;
      break;
    }
  }

  if (!updated_correctly) {
    fprintf(stderr, "✗ %s:\n  pair wasnt updated correctly.\n  expected but was not found: %.4f %u %u %.2f %.2f %.2f\n",
      test_case, 2., 0, 2, -1., 0., 1.);
    print_heap(heap);
    exit(-2);
  }

  heap_free(heap);

  printf("✓ %s\n", test_case);
}

void test_update_face() {
  const char* test_case = "update face";
  unsigned int i;

  double positions[] = {
    0., 0., 0.,
    1., 1., 1.,
    2., 1., 0.,
    0., -1., 0.,
    -1., 1., -1,
    -2., -1., 0.
  };

  unsigned int face[] = {
    0, 1, 2,
    5, 1, 3,
    1, 3, 4,
    5, 3, 4
  };

  bool *deleted_faces = calloc(4, sizeof(bool));

  bool expected[] = {
    false, true, true, false
  };

  Mesh m = {positions, NULL, face, 6, 4, 0};

  update_face(&m, deleted_faces, 1, 3);

  for (i = 0; i < 4; i++) {
    if (deleted_faces[i] != expected[i]) {
      fprintf(stderr, "✗ %s:\n  expected: %u, %u, %u, %u\n  got: %u, %u, %u, %u\n",
        test_case, false, true, true, false,
        deleted_faces[0], deleted_faces[1], deleted_faces[2], deleted_faces[3]);
      exit(-2);
    }
  }

  free(deleted_faces);

  printf("✓ %s\n", test_case);
}

void test_update_features() {
  const char* test_case = "update features";
  unsigned int i, j;

  double positions[] = {
    0., 0., 0.,
    1., 1., 1.,
    2., 1., 0.,
    0., -1., 0.,
    -1., 1., -1,
    -2., -1., 0.
  };

  double features[] = {
    0, 1, 2,
    5, 1, 3,
    1, 3, 4,
    5, 3, 4
  };

  double expected[] = {
    0, 1, 2,
    -1, -2, -3,
    1, 3, 4,
    5, 3, 4 // features are removed at the end
  };

  Pair* p1 = pair_init(3);
  p1->error = 1;
  p1->v1 = 1;
  p1->v2 = 3;
  p1->target[0] = 9;
  p1->target[1] = 9;
  p1->target[2] = 9;
  p1->feature[0] = -1;
  p1->feature[1] = -2;
  p1->feature[2] = -3;

  Mesh m = {positions, features, NULL, 6, 4, 3};

  update_features(&m, p1);

  for (i = 0; i < 12; i++) {
    if (m.features[i] != expected[i]) {
      fprintf(stderr, "✗ %s:\n  expected:                   got:\n", test_case);
      for (j = 0; j < 4; j++) {
        fprintf(stderr, " % .4f % .4f % .4f     % .4f % .4f % .4f\n",
        expected[j * 3], expected[j * 3 + 1], expected[j * 3 + 2],
        m.features[j * 3], m.features[j * 3 + 1], m.features[j * 3 + 2]
        );
      }
      exit(-2);
    }
  }

  pair_free(p1);

  printf("✓ %s\n", test_case);
}

int main(void) {
  test_update_pairs();
  test_update_face();
  test_update_features();
}