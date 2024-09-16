#!/bin/bash

make clean
make debug=${DEBUG}
exec ./main