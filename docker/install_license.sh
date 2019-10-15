#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "`basename "$0"` /path/to/license.lic"
    exit 1
fi

docker cp "$1" inendi-inspector:/opt/inendi/inspector.lic
