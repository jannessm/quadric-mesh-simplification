referenced_files = test_utils.c maths.c sparse_mat.c edges.c array.c pair.c pair_heap.c -lm -g

build: array.c clean_mesh.c contract_pair.c edges.c maths.c mesh_inversion.c pair.c pair_heap.c preserve_bounds.c q.c simplify.c sparse_mat.c targets.c test_utils.c utils.c valid_pairs.c ../simplify.pyx
	python ../../setup.py build_ext --inplace

test: test_Q test_preserve_bounds test_valid_pairs test_pair_heap test_contract_pair test_mesh_inversion

test_all: build
	python ../simplify_test.py

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

test_pair_heap:
	gcc pair_heap_test.c $(referenced_files) -o pair_heap_test.o 
	valgrind --leak-check=yes -q ./pair_heap_test.o

test_contract_pair: test_contract_pair_omp
	gcc contract_pair_test.c contract_pair.c targets.c $(referenced_files) -o contract_pair_test.o 
	valgrind --leak-check=yes -q ./contract_pair_test.o

test_contract_pair_omp:
	gcc -fopenmp contract_pair_test.c contract_pair.c targets.c $(referenced_files) -o contract_pair_test.o 
	./contract_pair_test.o

test_mesh_inversion: test_mesh_inversion_omp
	gcc mesh_inversion_test.c mesh_inversion.c $(referenced_files) -o mesh_inversion_test.o 
	valgrind --leak-check=yes -q ./mesh_inversion_test.o

test_mesh_inversion_omp:
	gcc -fopenmp mesh_inversion_test.c mesh_inversion.c $(referenced_files) -o mesh_inversion_test.o 
	./mesh_inversion_test.o