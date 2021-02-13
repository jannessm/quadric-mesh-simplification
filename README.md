# Quadric mesh simplification
A leightweight package for simplifying a mesh containing node features. The algorithm from [Surface Simplification Using Quadric Error Metrics](http://mgarland.org/files/papers/quadrics.pdf) was implemented using cython.

Only python versions >= 3.6 are supported

## Installation

### with pip
```bash
$ pip install quad_mesh_simplify
```

### from source if distribution is not supported
Download this repository and build the package by running:

```bash
$ pip install -r requirements.txt
$ python setup.py build_ext --inplace
$ pip install .
```

## Usage

This package provides one simple function to reduce a given mesh. This can be done for simple meshes or meshes with vertex features.

##### simplify_mesh(positions, face, num_nodes, features=None, threshold=0., max_err=np.Infinity)

`positions` (numpy array): array of shape [num_nodes x 3] containing the x, y, z position for each node

`face` (numpy array): array of shape [num_faces x 3] containing the indices for each triangular face

`num_nodes` (int): number of nodes that the final mesh will have
        threshold (number, optional): threshold of vertices distance to be a valid pair

`features` (numpy array): features for all nodes [num_nodes x feature_length]

`threshold` (double): if the distance between two vertices is below this threshold, they are considered as valid pairs that can be merged.

`max_err` (double): no vertices are merged that have an error higher than this number. IMPORTANT: if provided it is not guaranteed that the output will have less than num_nodes vertices.

Returns: `new_positions, new_face, (new_features)`

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

