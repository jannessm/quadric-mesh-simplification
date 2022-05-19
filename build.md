# Build package

1. bump2version (patch/minor/major)
2. `docker-compose up` or `docker run -v $(pwd):/src --entrypoint /src/dist.sh quay.io/pypa/manylinux2010_x86_64`
3. twine upload dist/quad*-manylinux*
