#!/bin/bash

while true; do
    # Run INENDI Inspector with "--devel" option to be able to generate crash reports
    output=`flatpak run --devel com.esi_inendi.Inspector 2>&1`
    ret=$?
    if [ ${ret} -ne 0 ] && [ ${ret} -ne 139 ] # 139 means segfault
    then
        output=`echo "${output}" | tail -n 10 `
        kdialog --error "${output}"
    fi
done
