#!/bin/bash
sudo opcontrol --start 2>&1 1>/dev/null
for i in $(seq 0 16 1024)
do
	sudo opcontrol --reset 2>&1 1>/dev/null
	echo -n "$i "
	./nred $((i*1024)) 5000000 2>&1 1>/dev/null
	sudo opcontrol --dump 2>&1 1>/dev/null
	opreport --no-header ./nred 2>/dev/null |tail -n+2 |head -n1 |awk ' { print $1 } '
done
