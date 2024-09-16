#! /usr/bin/env bash

mkdir -p ../results

DEBUG=${DEBUG:-0}
docker run --rm --name ImgCompression \
    -v $(pwd)/../results:/opt/ImgCompression/results \
    -e DEBUG=$DEBUG \
    pca:v0.3