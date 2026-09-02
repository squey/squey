#!/bin/bash
#
# Prints the devcontainer.json fragment that gives the container access to the
# host GPU, for the driver this host actually runs.
#
# It cannot be committed as-is: the flatpak runtime holding the NVIDIA userspace
# is named after the driver version, so the paths differ from one machine to the
# next. This is the same detection buildstream/.common.sh performs before
# handing the directories to "bst shell".
#
# Squey renders through OpenCL, and the sysroot has no usable platform of its
# own -- its PortableCL is built without ICD support -- so without this the
# container falls back to the CPU, which is what FORCE_CPU in devcontainer.json
# selects.

set -e

GL_RUNTIME="runtime/org.freedesktop.Platform.GL.default/x86_64/25.08/active/files"
GL_EXTRA_RUNTIME="runtime/org.freedesktop.Platform.GL.default/x86_64/25.08-extra/active/files"

for root in "$HOME/.local/share/flatpak" /var/lib/flatpak; do
    if [ -d "$root/$GL_RUNTIME" ]; then
        FLATPAK_ROOT="$root"
        break
    fi
done

if [ -z "$FLATPAK_ROOT" ]; then
    echo >&2 "No org.freedesktop.Platform.GL.default runtime found."
    echo >&2 "Install it with: flatpak install flathub org.freedesktop.Platform.GL.default//25.08"
    exit 1
fi

DRIVER="$(flatpak --gl-drivers 2>/dev/null | grep '^nvidia' | head -1)"
if [ -z "$DRIVER" ]; then
    echo >&2 "No NVIDIA flatpak driver reported by 'flatpak --gl-drivers'."
    echo >&2 "Only NVIDIA is wired up here; on any other GPU, passing /dev/dri alone"
    echo >&2 "gives OpenGL but no OpenCL, and Squey needs OpenCL to draw."
    exit 1
fi

NVIDIA_RUNTIME="runtime/org.freedesktop.Platform.GL.$DRIVER/x86_64/1.4/active/files"
if [ ! -d "$FLATPAK_ROOT/$NVIDIA_RUNTIME" ]; then
    echo >&2 "Driver $DRIVER is in use but its flatpak runtime is missing."
    echo >&2 "Install it with: flatpak install flathub org.freedesktop.Platform.GL.$DRIVER"
    exit 1
fi

GL_TARGET_DIR="/usr/lib/x86_64-linux-gnu/GL"

cat << EOF
  "mounts": [
    "source=$FLATPAK_ROOT/$GL_RUNTIME,target=$GL_TARGET_DIR,type=bind,readonly",
    "source=$FLATPAK_ROOT/$GL_EXTRA_RUNTIME,target=$GL_TARGET_DIR/default,type=bind,readonly",
    "source=$FLATPAK_ROOT/$NVIDIA_RUNTIME,target=$GL_TARGET_DIR/$DRIVER,type=bind,readonly"
  ],
  "containerEnv": {
    // run_cmd.sh links libnvidia-ptxjitcompiler into the ICD directory, but the
    // OpenCL compiler dlopens libnvidia-nvvm.so.4 as well, and finds it nowhere
    // unless the runtime directory itself is on the search path. Leaving it out
    // is what makes clBuildProgram fail with CL_BUILD_PROGRAM_FAILURE (-11)
    // after the device has been found and reported.
    "LD_LIBRARY_PATH": "$GL_TARGET_DIR/$DRIVER/lib:$GL_TARGET_DIR/lib:/usr/lib/x86_64-linux-gnu:/app/lib"
  },
  "runArgs": [
    "--userns=keep-id",
    "--device=/dev/nvidia0", "--device=/dev/nvidiactl", "--device=/dev/nvidia-uvm",
    "--device=/dev/dri"
  ]
EOF

echo >&2
echo >&2 "Merge the above into .devcontainer/devcontainer.json, and drop the"
echo >&2 "FORCE_CPU entry of containerEnv so that the GPU is actually used."
