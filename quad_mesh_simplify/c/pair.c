#include <stdlib.h>
#include "pair.h"

PairList* pairlist_init(void) {
  PairList* list = malloc(sizeof(PairList));
  list->length = 0;
  list->list = malloc(sizeof(Pair*) * 0);
  return list;
}

void pairlist_append(PairList* list, Pair* pair) {
  list->length++;
  list->list = realloc(list->list, sizeof(Pair*) * list->length);
  list->list[list->length - 1] = pair;
}

void pairlist_free(PairList* list) {
  unsigned int i;
  for (i = 0; i < list->length; i++) {
    pair_free(list->list[i]);
  }
  free(list->list);
  free(list);
}

Pair* pair_init(unsigned int feature_length) {
  Pair* new_pair = malloc(sizeof(Pair));
  new_pair->feature = malloc(sizeof(double) * feature_length);
  return new_pair;
}

void pair_free(Pair* pair) {
  free(pair->feature);
  free(pair);
}