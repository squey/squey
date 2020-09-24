#!/bin/bash

while true; do
    # Run INENDI Inspector with "--devel" option to be able to generate crash reports
    branch=`cat /opt/inendi/inspector/current_branch.txt`
    [[ -z "$branch" ]] && b="master" || b="$branch"
    PYTHON_VERSION=`flatpak run --command="python3" com.esi_inendi.Inspector//$b -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))'`
    output=`flatpak run --devel --env="PYTHONPATH=/opt/inendi/inspector/python${PYTHON_VERSION}/site-packages" com.esi_inendi.Inspector//$b 2>&1`
    ret=$?
    if [ ${ret} -ne 0 ] && [ ${ret} -ne 139 ] # 139 means segfault
    then
        output=`echo "${output}" | tail -n 10 `
        kdialog --error "${output}"
    fi
done