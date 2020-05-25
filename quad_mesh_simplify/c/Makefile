test: test_Q test_preserve_bounds

test_Q:
	gcc -fopenmp q_test.c q.c test_utils.c maths.c -o q_test.o -lm -g
	valgrind --leak-check=yes -q ./q_test.o

test_preserve_bounds:
	gcc -fopenmp preserve_bounds_test.c preserve_bounds.c test_utils.c maths.c sparse_mat.c -o preserve_bounds_test.o -lm -g
	valgrind --leak-check=yes -q ./preserve_bounds_test.o