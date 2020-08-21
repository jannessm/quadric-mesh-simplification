#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "pair_heap.h"
#include "pair.h"
#include "test_utils.h"

const char* test_case = "build heap";

int main(void) {

  Pair* p1 = pair_init(0);
  p1->error = 5;
  p1->v1 = 1;
  p1->v2 = 2;


  Pair* p2 = pair_init(0);
  p2->error = 0;
  p2->v1 = 3;
  p2->v2 = 2;


  Pair* p3 = pair_init(0);
  p3->error = 2;
  p3->v1 = 1;
  p3->v2 = 3;

  Pair* p4 = pair_init(0);
  p4->error = 3;
  p4->v1 = 1;
  p4->v2 = 0;

  PairList* list = pairlist_init();
  pairlist_append(list, p1);
  pairlist_append(list, p2);
  pairlist_append(list, p3);
  pairlist_append(list, p4);

  PairHeap* heap = list_to_heap(list);

  if (heap_get_error(heap, 1) != 0.0) {
    fprintf(stderr, "✗ %s:\n  Heap was not initialized correctly! Got:\n", test_case);
    print_heap(heap);
  }

  heap_free(heap);

  printf("✓ %s\n", test_case);
}