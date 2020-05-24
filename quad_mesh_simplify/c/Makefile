test: test_Q test_preserve_bounds

test_Q:
	gcc-9 -fopenmp q_test.c q.c test_utils.c utils.c maths.c -o q_test.o
	./q_test.o

test_preserve_bounds:
	gcc-9 -fopenmp preserve_bounds_test.c preserve_bounds.c utils.c test_utils.c maths.c -o preserve_bounds_test.o
	./preserve_bounds_test.o