#!/bin/bash

GL_TARGET_DIR="/usr/lib/x86_64-linux-gnu/GL"
# /etc/opencl_vendors is where the flatpak sandbox expects the vendor files, and
# it is writable there. It is not in a BuildStream build sandbox, where /etc is
# read only -- so the testsuite ran with no OpenCL platform at all, silently,
# neither the mkdir nor the redirections below being checked. Honour a caller
# that already named a directory, and fall back to a writable one rather than
# carry on writing nowhere.
if [ -z "$OCL_ICD_VENDORS" ]; then
	export OCL_ICD_VENDORS=/etc/opencl_vendors
fi
if ! mkdir -p "$OCL_ICD_VENDORS" 2> /dev/null; then
	export OCL_ICD_VENDORS="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/opencl_vendors"
	mkdir -p "$OCL_ICD_VENDORS"
fi

NVIDIA_VERSION_NAME=`ls $GL_TARGET_DIR|grep "nvidia-*"|sed -e "s/nvidia-//"`
NVIDIA_VERSION=`echo $NVIDIA_VERSION_NAME | sed 's/-/./g'`

# export PYTHONPATH
PYTHON_VERSION=`python3 -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))'`
export PYTHONPATH="$PYTHONPATH:$(echo $XDG_DATA_HOME/python/lib/python${PYTHON_VERSION}/site-packages)"

# export LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH

echo "/app/lib/libpocl.so" > $OCL_ICD_VENDORS/pocl.icd
if [ -n "$NVIDIA_VERSION" ]; then
	echo "$GL_TARGET_DIR/nvidia-$NVIDIA_VERSION_NAME/lib/libnvidia-opencl.so.$NVIDIA_VERSION" > $OCL_ICD_VENDORS/nvidia.icd
	# The OpenCL implementation dlopens its compiler at the first clBuildProgram,
	# and finds it by soname alone: libnvidia-ptxjitcompiler.so.1, and
	# libnvidia-nvvm.so.4 with it since the 5xx drivers. Missing either one, the
	# device is found and reported, then every kernel build fails with
	# CL_BUILD_PROGRAM_FAILURE (-11). Rather than name them one by one and go
	# through this again on the next driver generation, publish every soname
	# symlink the runtime carries; they are all libnvidia-*, so nothing here can
	# shadow a library of the sysroot.
	NVIDIA_EXTRA_LIBS_PATH=$OCL_ICD_VENDORS/lib
	mkdir -p $NVIDIA_EXTRA_LIBS_PATH
	for nvidia_lib in $GL_TARGET_DIR/nvidia-$NVIDIA_VERSION_NAME/lib/libnvidia-*.so.*; do
		[ -e "$nvidia_lib" ] || continue
		ln -sf "$nvidia_lib" "$NVIDIA_EXTRA_LIBS_PATH/$(basename "$nvidia_lib")"
	done
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

"$@"
