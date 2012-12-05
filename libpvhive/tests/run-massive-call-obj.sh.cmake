#!/bin/sh

run()
{
    ./Thive_massive_call_obj $@
}

run 10000000 1       0
run 10000    1000    0
run 10       1000000 0

run 10000000 1       1
run 10000    1000    1
run 10       1000000 1

run 1000000 1       1000
run 1000    1000    1000
run 1       1000000 1000

run 1000    1    1000000
run 1       1000 1000000


rm -f log.$$
