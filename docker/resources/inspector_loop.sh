#!/bin/bash

while true; do
    # Run INENDI Inspector with "--devel" option to be able to generate crash reports
    flatpak run --devel com.esi_inendi.Inspector &

    # Disable coredumpctl core dump handling
    pid="$!"
    echo "0x0" > /proc/$pid/coredump_filter
    wait $pid
done
