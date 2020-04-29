# Quadric mesh simplification
A leightweight package for simplifying a mesh containing node features. The algorithm from [Surface Simplification Using Quadric Error Metrics](http://mgarland.org/files/papers/quadrics.pdf) was implemented using cython.

## Installation

```bash
$ python setup.py build_ext --inplace
```

## Usage

This package provides one simple function to reduce a given mesh

```python
    from quad_mesh_simplify import simplify_mesh
```

To reduce a mesh call

```python
    new_positions, new_face = simplify_mesh(positions, face, <final_num_nodes>)
```

