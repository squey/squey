#!/bin/sh

run()
{
    exec ./Tmassive-call-prop $@ | tee log.$$
    OPS=`grep 'ops per sec' log.$$ | sed -e 's/^.*: //'`
    echo "@ $@ $OPS"
}

run 1000000000 1       0
run 1000000    1000    0
run 1000       1000000 0

run 1000000000 1       1
run 1000000    1000    1
run 1000       1000000 1

run 1000000 1       1000
run 1000    1000    1000
run 10      100000  1000

run 1000    1    1000000
run 10      100  1000000


rm -f log.$$
