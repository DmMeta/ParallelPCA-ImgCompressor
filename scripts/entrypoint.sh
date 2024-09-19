#! /usr/bin/env bash

make clean
make debug=${DEBUG} simd=${SIMD}
exec ./main