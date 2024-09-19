#! /usr/bin/env bash

mkdir -p ../results
# Run the application
docker run --rm --name ImgCompression \
    -v $(pwd)/../results:/opt/ImgCompression/results \
    -e DEBUG=1 \
    -e SIMD=0 \
    pca:v0.4

pytest ../tests/

