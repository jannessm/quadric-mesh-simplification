import array

cdef class PairHeap:

    def __cinit__(self, double [:, :] pairs):
        self.nodes_ = array.array('I', [0] + list(range(pairs.shape[0])))
        self.nodes = self.nodes_
        self.pairs = pairs
        self.length_ = self.nodes.shape[0]
        self.build()
    
    cpdef long length(self):
        return self.length_ - 1

    cdef double _get_value(self, long i):
        return self.pairs[self.nodes[i], 0]

    cdef double get_value(self, long i):
        return self.pairs[self.nodes[i + 1], 0]

    cpdef double[:] get_pair(self, long i):
        return self.pairs[self.nodes[i + 1]]

    cdef void insert(self, long i):
        cdef array.array new = array.array('I', [i])
        array.extend(self.nodes_, new)
        self.nodes = self.nodes_

        self.percolate_up(self.length_)
        
        self.length_ += 1

    cdef void build(self):
        cdef long i = self.length_ // 2
        while i > 0:
            self.percolate_down(i)
            i -= 1

    cdef double[:] pop(self):
        cdef double[:] root = self.pairs[self.nodes[1]]
        self.nodes[1] = self.nodes[self.length_ - 1]
        
        array.resize_smart(self.nodes_, self.length_)
        self.length_ -= 1
        
        self.nodes = self.nodes_

        self.percolate_down(1)
        return root

    cdef void percolate_up(self, long i):
        cdef long tmp
        
        while i // 2 > 0:
            if self._get_value(i) < self._get_value(i // 2):
                tmp = self.nodes[i // 2]
                self.nodes[i // 2] = self.nodes[i]
                self.nodes[i] = tmp
            i = i // 2

    cdef void percolate_down(self, long i):
        cdef long tmp, min_child
        
        while i * 2 < self.length_:
            min_child = self.get_min_child(i)
            if self._get_value(i) > self._get_value(min_child):
                tmp = self.nodes[i]
                self.nodes[i] = self.nodes[min_child]
                self.nodes[min_child] = tmp
            i = min_child

    cdef long get_min_child(self, long i):
        if i * 2 + 1 >= self.length_:
            return i * 2
        elif self._get_value(i * 2) < self._get_value(i * 2 + 1):
            return i * 2
        else:
            return i * 2 + 1

    def __str__(self):
        return '+- {}\n|   +- {}\n|   +- {}\n'.format(
            self._get_value(1),
            self.__node_repr(2, 1),
            self.__node_repr(3, 1))

    def __node_repr(self, i, level):
        if self.length_ - 1 < i:
            return None
        
        return '{}\n{}+- {}\n{}+- {}'.format(
            self._get_value(i),
            '|   '*(level + 1),
            self.__node_repr(i * 2, level + 1),
            '|   '*(level + 1),
            self.__node_repr(i * 2 + 1, level + 1)
        )
