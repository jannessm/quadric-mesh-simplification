referenced_files = test_utils.c maths.c sparse_mat.c edges.c -lm -g

test: test_Q test_preserve_bounds

test_Q:
	gcc -fopenmp q_test.c q.c $(referenced_files) -o q_test.o 
	valgrind --leak-check=yes -q ./q_test.o

test_preserve_bounds:
	gcc -fopenmp preserve_bounds_test.c preserve_bounds.c $(referenced_files) -o preserve_bounds_test.o -lm -g
	valgrind --leak-check=yes -q ./preserve_bounds_test.o

test_valid_pairs:
	gcc -fopenmp valid_pairs_test.c valid_pairs.c $(referenced_files) -o valid_pairs_test.o -lm -g
	valgrind --leak-check=yes -q ./valid_pairs_test.o
