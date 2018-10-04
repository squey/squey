#!/bin/bash

if [ $# -ne 1 ]; then
	echo "usage : `basename $0` <appdata_path_linux>"
    exit 1
fi

inspector_config_dir="$1/Inspector"
mkdir -p "$inspector_config_dir" &> /dev/null
cd /home/inendi
unlink /home/inendi/.inendi &> /dev/null
ln -s "$inspector_config_dir" /home/inendi/.inendi &> /dev/null