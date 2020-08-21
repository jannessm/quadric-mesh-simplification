#include <stdbool.h>
#include "pair_heap.h"
#include "mesh.h"

void update_pairs(
    PairHeap* heap,
    Mesh* mesh,
    double* Q,
    unsigned int v1,
    unsigned int v2);

void update_face(
    Mesh* mesh,
    bool* deleted_faces,
    unsigned int v1,
    unsigned int v2);

void update_features(
    Mesh* mesh,
    Pair* pair);