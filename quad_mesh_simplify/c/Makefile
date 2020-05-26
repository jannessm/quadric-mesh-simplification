referenced_files = test_utils.c maths.c sparse_mat.c edges.c array.c pair.c -lm -g

test: test_Q test_preserve_bounds test_valid_pairs

test_Q: test_Q_omp
	gcc q_test.c q.c $(referenced_files) -o q_test.o 
	valgrind --leak-check=yes -q ./q_test.o

test_Q_omp:
	gcc -fopenmp q_test.c q.c $(referenced_files) -o q_test.o 
	./q_test.o

test_preserve_bounds: test_preserve_bounds_omp
	gcc preserve_bounds_test.c preserve_bounds.c $(referenced_files) -o preserve_bounds_test.o -lm -g
	valgrind --leak-check=yes -q ./preserve_bounds_test.o

test_preserve_bounds_omp:
	gcc -fopenmp preserve_bounds_test.c preserve_bounds.c $(referenced_files) -o preserve_bounds_test.o -lm -g
	./preserve_bounds_test.o

test_valid_pairs: test_valid_pairs_omp
	gcc valid_pairs_test.c valid_pairs.c $(referenced_files) -o valid_pairs_test.o -lm -g
	valgrind --leak-check=yes -q ./valid_pairs_test.o

test_valid_pairs_omp: 
	gcc -fopenmp valid_pairs_test.c valid_pairs.c $(referenced_files) -o valid_pairs_test.o -lm -g
	./valid_pairs_test.o

test_targets: test_targets_omp
	gcc targets_test.c targets.c $(referenced_files) -o targets_test.o 
	valgrind --leak-check=yes -q ./targets_test.o

test_targets_omp:
	gcc -fopenmp targets_test.c targets.c $(referenced_files) -o targets_test.o 
	./targets_test.o
