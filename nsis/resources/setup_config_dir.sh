#!/bin/sh

if [ $# -ne 1 ]; then
	echo "usage : `basename $0` <appdata_path_linux>"
    exit 1
fi

squey_config_dir="$1/Squey"
mkdir -p "$squey_config_dir" &> /dev/null
cd /home/squey
unlink /home/squey/.squey &> /dev/null
ln -s "$squey_config_dir" /home/squey/.squey &> /dev/null