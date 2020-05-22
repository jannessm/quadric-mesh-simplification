#ifndef PAIR_HEAP
#define PAIR_HEAP
#include "pair_heap.c"
#endif

void update_pairs(
    unsigned int v1,
    unsigned int v2,
    PairHeap heap,
    double** positions,
    double*** Q,
    double** features);

void update_face(
    unsigned int v1,
    unsigned int v2,
    unsigned int** face,
    char* deleted_faces);

void update_features(
    double* pair,
    double** features);