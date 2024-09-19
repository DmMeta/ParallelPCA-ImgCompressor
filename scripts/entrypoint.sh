#! /usr/bin/env bash

make clean
make debug=${DEBUG}
exec ./main