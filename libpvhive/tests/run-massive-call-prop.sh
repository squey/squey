#!/bin/sh

run()
{
    ./bin/Thive_massive_call_prop $@ | tee log.$$
}

run 10000000 1       0
run 10000    1000    0
run 10       1000000 0

run 10000000 1       1
run 10000    1000    1
run 10       1000000 1

run 1000000 1       1000
run 1000    1000    1000
run 10      100000  1000

run 1000    1    1000000
run 10      100  1000000
