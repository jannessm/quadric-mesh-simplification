#include <stdlib.h>
#include <stdio.h>
#include "pair_heap.h"

PairHeap* list_to_heap(PairList* pairs) {
  unsigned int i;
  
  PairHeap* heap = malloc(sizeof(PairHeap));
  heap->length = pairs->length + 1;
  // first empty is used to fit arithmetic
  heap->nodes = malloc((pairs->length + 1) * sizeof(Pair*));

  for (i = 1; i < heap->length; i++) {
    heap->nodes[i] = pairs->list[i - 1];
  }

  free(pairs->list);
  free(pairs);

  heap_build(heap);
  return heap;
}

double heap_get_error(PairHeap* heap, unsigned int i) {
  return (heap_get_pair(heap, i)->error);
}

Pair* heap_get_pair(PairHeap* heap, unsigned int i) {
  return heap->nodes[i];
}

void heap_build(PairHeap* heap) {
  unsigned int i;
  for (i = heap->length >> 1; i > 0; i--) {
    heap_percolate_down(heap, i);
  }
}

Pair* heap_pop(PairHeap* heap) {
  Pair* root = heap->nodes[1];
  heap->nodes[1] = heap->nodes[heap->length - 1];
  heap->length--;
  heap->nodes = realloc(heap->nodes, heap->length * sizeof(Pair*));
  heap_percolate_down(heap, 1);
  return root;
}

void heap_percolate_down(PairHeap* heap, unsigned int i) {
  unsigned int min_child;
  Pair *tmp;

  while (i * 2 < heap->length) {
    min_child = get_min_child(heap, i);

    if (heap_get_error(heap, i) > heap_get_error(heap, min_child)) {
      tmp = heap->nodes[i];
      heap->nodes[i] = heap->nodes[min_child];
      heap->nodes[min_child] = tmp;
    }

    i = min_child;
  }
}

unsigned int get_min_child(PairHeap* heap, unsigned int i) {
  if (i * 2 + 1 >= heap->length) {
    return i * 2;
  } else if (heap_get_error(heap, i * 2) < heap_get_error(heap, i * 2 + 1)) {
    return i * 2;
  } else {
    return i * 2 + 1;
  }
}

void heap_free(PairHeap* heap) {
  unsigned int i;
  for (i = 1; i < heap->length; i++) {
    pair_free(heap->nodes[i]);
  }
  free(heap->nodes);
  free(heap);
}

void print_node(PairHeap* heap, unsigned int i, unsigned int level) {
  unsigned int j;
  if (heap->length - 1 < i) {
    printf("None");
  } else {
    printf("%f (%u, %u)\n",
      heap_get_error(heap, i),
      heap_get_pair(heap, i)->v1,
      heap_get_pair(heap, i)->v2);

    for (j = 0; j < level + 1; j++) {
      printf("|   ");
    }
    printf("+- ");
    print_node(heap, i * 2, level + 1);
    printf("\n");

    for (j = 0; j < level + 1; j++) {
      printf("|   ");
    }
    printf("+- ");
    print_node(heap, i * 2 + 1, level + 1);
  }
}

void print_heap(PairHeap* heap) {
  printf("+- %f (%u, %u)\n|   +- ",
    heap_get_error(heap, 1),
    heap_get_pair(heap, 1)->v1,
    heap_get_pair(heap, 1)->v2
  );
  print_node(heap, 2, 1);
  printf("\n|   +- ");
  print_node(heap, 3, 1);
  printf("\n");
}
