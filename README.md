# Quadric mesh simplification
A leightweight package for simplifying a mesh containing node features. The algorithm from [Surface Simplification Using Quadric Error Metrics](http://mgarland.org/files/papers/quadrics.pdf) was implemented using cython.

## Installation

Download this repository and build the package by running:

```bash
$ python setup.py build_ext --inplace
```

## Usage

This package provides one simple function to reduce a given mesh. This can be done for simple meshes or meshes with vertex features.

### Reduce a simple mesh

```python
    from quad_mesh_simplify import simplify_mesh

    new_positions, new_face = simplify_mesh(positions, face, <final_num_nodes>)
```

### Reduce a mesh with vertex features
```python
    from quad_mesh_simplify import simplify_mesh

    new_positions, new_face = simplify_mesh(positions, face, <final_num_nodes>, features=features)
```

### Reduce a mesh with a threshold for the minimal distance

```python
    from quad_mesh_simplify import simplify_mesh

    new_positions, new_face = simplify_mesh(positions, face, <final_num_nodes>, threshold=0.5)
```
