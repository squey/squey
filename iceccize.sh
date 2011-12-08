#!/bin/bash
rm CMakeCache.txt
#make clean
CC=icecc CXX=icecc cmake .
