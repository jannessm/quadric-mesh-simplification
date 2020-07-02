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
  new_pair->faces = malloc(sizeof(unsigned int) * 0);
  new_pair->n_faces = 0;
  return new_pair;
}

void pair_add_face(Pair* pair, unsigned int face_id) {
  pair->n_faces++;
  pair->faces = realloc(pair->faces, sizeof(unsigned int) * pair->n_faces);
  pair->faces[pair->n_faces - 1] = face_id;
}

void pair_free(Pair* pair) {
  free(pair->feature);
  free(pair->faces);
  free(pair);
}