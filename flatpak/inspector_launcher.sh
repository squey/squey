#!/bin/bash

mkdir -p /etc/inendi/licenses
cp ~/.inendi/licenses/inendi-inspector.lic /etc/inendi/licenses/inendi-inspector.lic >/dev/null

inendi-lsinit &> /dev/null

NVIDIA_VERSION=`ls /usr/lib/GL|grep nvidia-*|sed -rn 's/nvidia-([[:digit:]]+)-([[:digit:]]+)/\1\.\2/p'`

mkdir -p /etc/OpenCL/vendors
echo "/app/lib/libpocl.so" > /etc/OpenCL/vendors/pocl.icd
if [ -n "$NVIDIA_VERSION" ]; then
	echo "libnvidia-opencl.so.$NVIDIA_VERSION" > /etc/OpenCL/vendors/nvidia.icd
fi

inendi-inspector
