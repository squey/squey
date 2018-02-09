#!/bin/bash

inendi-lsinit &> /dev/null

NVIDIA_VERSION_NAME=`ls /usr/lib/GL|grep nvidia-*|sed -e "s/nvidia-//"`
NVIDIA_VERSION=`echo $NVIDIA_VERSION_NAME | sed -rn 's/([[:digit:]]+)-([[:digit:]]+)/\1\.\2/p'`

mkdir -p /etc/OpenCL/vendors
echo "/app/lib/libpocl.so" > /etc/OpenCL/vendors/pocl.icd
if [ -n "$NVIDIA_VERSION" ]; then
	echo "/usr/lib/GL/nvidia-$NVIDIA_VERSION_NAME/lib/libnvidia-opencl.so.$NVIDIA_VERSION" > /etc/OpenCL/vendors/nvidia.icd
fi

inendi-inspector
