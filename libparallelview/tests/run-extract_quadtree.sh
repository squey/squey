#!/bin/sh

TEST_NUM=`./extract_quadtree 1 -1`
for COUNT in 1000 10000 100000 1000000 10000000 100000000 ; do
    echo "##############################################################################"
    echo "## COUNT = $COUNT"
    for ALGO in `seq 0 $TEST_NUM` ; do
	./extract_quadtree $COUNT $ALGO
    done
done | tee log.extract_quadtree.`hostname`
