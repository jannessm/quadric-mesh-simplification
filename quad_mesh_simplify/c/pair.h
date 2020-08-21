#ifndef INCLUDE_PAIR
#define INCLUDE_PAIR
typedef struct Pair {
  double error;
  unsigned int v1;
  unsigned int v2;
  double target[3];
  double* feature;
  unsigned int* faces;
  unsigned int n_faces;
} Pair;

typedef struct PairList {
  unsigned int length;
  Pair** list;
} PairList;
#endif

PairList* pairlist_init(void);

void pairlist_append(PairList* list, Pair* pair);

void pairlist_free(PairList* list);

Pair* pair_init(unsigned int feature_length);

void pair_add_face(Pair* pair, unsigned int face_id);

void pair_free(Pair* pair);