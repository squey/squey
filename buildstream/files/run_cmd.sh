#!/bin/bash

inendi-lsinit &> /dev/null

GL_TARGET_DIR="/usr/lib/x86_64-linux-gnu/GL"
export OCL_ICD_VENDORS=/etc/opencl_vendors
mkdir -p $OCL_ICD_VENDORS

NVIDIA_VERSION_NAME=`ls $GL_TARGET_DIR|grep "nvidia-*"|sed -e "s/nvidia-//"`
NVIDIA_VERSION=`echo $NVIDIA_VERSION_NAME | sed 's/-/./g'`

echo "/app/lib/libpocl.so" > $OCL_ICD_VENDORS/pocl.icd
if [ -n "$NVIDIA_VERSION" ]; then
	echo "$GL_TARGET_DIR/nvidia-$NVIDIA_VERSION_NAME/lib/libnvidia-opencl.so.$NVIDIA_VERSION" > $OCL_ICD_VENDORS/nvidia.icd
	NVIDIA_EXTRA_LIBS_PATH=$OCL_ICD_VENDORS/lib
	mkdir $NVIDIA_EXTRA_LIBS_PATH
	ln -s $GL_TARGET_DIR/nvidia-$NVIDIA_VERSION_NAME/lib/libnvidia-ptxjitcompiler.so.$NVIDIA_VERSION $NVIDIA_EXTRA_LIBS_PATH/libnvidia-ptxjitcompiler.so.1
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVIDIA_EXTRA_LIBS_PATH
fi

# DCV compatibility
: "${DCV_GL_DIR:=/var/lib/dcv-gl/lib64}"
: "${DCV_GL_FLATPAK_DIR:=/var/lib/dcv-gl/flatpak}"
if [ -d ${DCV_GL_FLATPAK_DIR} ]; then
    mkdir -p "${DCV_GL_DIR}"
    ln -s "/usr/lib/libGL.so.1.0.0" "${DCV_GL_DIR}/libGL_SYS.so.1.0.0"
    export LD_PRELOAD="${DCV_GL_FLATPAK_DIR}/libGL_WRAPPER.so.1.0.0 ${DCV_GL_FLATPAK_DIR}/libGL_DCV.so $LD_PRELOAD"
    export LD_LIBRARY_PATH=${DCV_GL_FLATPAK_DIR}:$LD_LIBRARY_PATH
fi

eval $@

