#!/bin/bash

source /opt/squey/squey/update_flatpak_packages.sh

while true; do
    # Run Squey with "--allow=devel" option to be able to generate crash reports
    branch=`cat /opt/squey/squey/current_branch.txt`
    [[ -z "$branch" ]] && b="stable" || b="$branch"
    PYTHON_VERSION=`flatpak run --command="python3" org.squey.Squey//$b -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))'`
    output=`flatpak run --allow=devel --env="PYTHONPATH=/opt/squey/squey/python${PYTHON_VERSION}/site-packages" --env="DISABLE_FOLLOW_SYSTEM_THEME=true" org.squey.Squey//$b 2>&1`
    ret=$?
    if [ ${ret} -ne 0 ] && [ ${ret} -ne 139 ] # 139 means segfault
    then
        output=`echo "${output}" | tail -n 10 `
        kdialog --error "${output}"
    fi
done