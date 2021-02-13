# Build package

1. bump2version (patch/minor/major)
2. docker-compose up
3. twine check dist/*
4. twine upload dist/*