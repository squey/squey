#!/bin/bash
./configure
make distclean
./configure --prefix=/usr/local --assembly
make clean
make
