typedef struct {
  void** nodes;
  int length;
  double** pairs;
} PairHeap;

void heap_get_value(PairHeap heap, int i) {}

double* heap_get_pair(PairHeap heap, int i) {}

void heap_insert(PairHeap heap, int i) {}

void heap_build(PairHeap heap) {}

double* heap_pop(PairHeap heap) {}

void heap_percolate_up(PairHeap heap, int i) {}

void heap_percolate_down(PairHeap heap, int i) {}

int get_min_child(PairHeap heap, int i) {}