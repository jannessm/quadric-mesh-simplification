#!/bin/bash
/opt/python/cp36-cp36m/bin/pip install -r /src/requirements.txt
/opt/python/cp36-cp36m/bin/pip wheel /src -w /src/dist

/opt/python/cp37-cp37m/bin/pip install -r /src/requirements.txt
/opt/python/cp37-cp37m/bin/pip wheel /src -w /src/dist

/opt/python/cp38-cp38/bin/pip install -r /src/requirements.txt
/opt/python/cp38-cp38/bin/pip wheel /src -w /src/dist

/opt/python/cp39-cp39/bin/pip install -r /src/requirements.txt
/opt/python/cp39-cp39/bin/pip wheel /src -w /src/dist

auditwheel repair /src/dist/quad_mesh_simplify*cp36*whl -w /src/dist
auditwheel repair /src/dist/quad_mesh_simplify*cp37*whl -w /src/dist
auditwheel repair /src/dist/quad_mesh_simplify*cp38*whl -w /src/dist
auditwheel repair /src/dist/quad_mesh_simplify*cp39*whl -w /src/dist