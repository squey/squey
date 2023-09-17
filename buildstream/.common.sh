#!/bin/bash

export PATH="${PATH}:${HOME}/.local/bin"

command -v "pip3" &> /dev/null || { echo >&2 "'pip3' executable not found, please install python3-pip"; exit 1; }
command -v "flatpak" &> /dev/null || { echo >&2 "'flatpak' executable not found, please install Flatpak"; exit 1; }

source env.conf

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -z "$WORKSPACE_PREFIX" ]; then
    WORKSPACE_PREFIX="$DIR"
fi

GL_HOST_DIR="runtime/org.freedesktop.Platform.GL.default/x86_64/22.08/active/files"
GL_HOST_DIR_USER="$HOME/.local/share/flatpak/$GL_HOST_DIR"
GL_HOST_DIR_SYSTEM="/var/lib/flatpak/$GL_HOST_DIR"
GL_EXTRA_HOST_DIR="runtime/org.freedesktop.Platform.GL.default/x86_64/22.08-extra/active/files"
GL_EXTRA_HOST_DIR_USER="$HOME/.local/share/flatpak/$GL_EXTRA_HOST_DIR"
GL_EXTRA_HOST_DIR_SYSTEM="/var/lib/flatpak/$GL_EXTRA_HOST_DIR"
NVIDIA_VERSION_NAME=$(flatpak --gl-drivers|grep "nvidia") || true
NVIDIA_HOST_DIR="runtime/org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME/x86_64/1.4/active/files"
NVIDIA_HOST_DIR_USER="$HOME/.local/share/flatpak/$NVIDIA_HOST_DIR"
NVIDIA_HOST_DIR_SYSTEM="/var/lib/flatpak/$NVIDIA_HOST_DIR"

GL_MOUNT_OPTS=""
if [ -d "$GL_HOST_DIR_USER" ]; then
    GL_MOUNT_OPTS="--mount $GL_HOST_DIR_USER $GL_TARGET_DIR"
    GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $GL_EXTRA_HOST_DIR_USER $GL_TARGET_DIR/default"
elif [ -d "$GL_HOST_DIR_SYSTEM" ]; then
    GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $GL_HOST_DIR_SYSTEM $GL_TARGET_DIR"
    GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $GL_EXTRA_HOST_DIR_SYSTEM $GL_TARGET_DIR/default"
fi

if [ -z "$NVIDIA_VERSION_NAME" ]; then
    echo "Please, install NVIDIA Drivers in order to have GPU acceleration."
elif [ -d "$NVIDIA_HOST_DIR_USER" ]; then
    GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $NVIDIA_HOST_DIR_USER $GL_TARGET_DIR/$NVIDIA_VERSION_NAME"
elif [ -d "$NVIDIA_HOST_DIR_SYSTEM" ]; then
    GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $NVIDIA_HOST_DIR_SYSTEM $GL_TARGET_DIR/$NVIDIA_VERSION_NAME"
else
    echo "Please, install flatpaked NVIDIA Drivers in order to have GPU acceleration (flatpak install flathub org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME)"
fi

MOUNT_OPTS="$GL_MOUNT_OPTS --mount opencl_vendors /etc/opencl_vendors --mount /srv/tmp-squey /srv/tmp-squey"

# Install Buildstream and bst-external plugins if needed
command -v "bst" &> /dev/null || { pip install --user BuildStream==2.0.1; }
python3 -c "import bst_external" &> /dev/null || pip install --break-system-packages --user -e "$DIR/plugins/bst-external"
pip install --break-system-packages --user -e "$DIR/plugins/buildstream-plugins"
pip install --break-system-packages --user -e "$DIR/plugins/bst-plugins-experimental" && pip install --break-system-packages --user dulwich
