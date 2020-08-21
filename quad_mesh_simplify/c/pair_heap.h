#include "pair.h"

#ifndef INCLUDE_PAIR_HEAP
  #define INCLUDE_PAIR_HEAP
  typedef struct {
    Pair** nodes;
    unsigned int length;
  } PairHeap;
#endif


PairHeap* list_to_heap(PairList* pairs);

double heap_get_error(PairHeap* heap, unsigned int i);

Pair* heap_get_pair(PairHeap* heap, unsigned int i);

void heap_insert(PairHeap* heap, unsigned int i);

void heap_build(PairHeap* heap);

Pair* heap_pop(PairHeap* heap);

void heap_percolate_up(PairHeap* heap, unsigned int i);

void heap_percolate_down(PairHeap* heap, unsigned int i);

unsigned int get_min_child(PairHeap* heap, unsigned int i);

void print_heap(PairHeap* heap);

void heap_free(PairHeap* heap);