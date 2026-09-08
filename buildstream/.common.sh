#!/bin/bash

TARGET_TRIPLES="x86_64-linux-gnu x86_64-apple-darwin aarch64-apple-darwin"
echo "$TARGET_TRIPLES" | tr " " '\n' | grep -F -q -x "$TARGET_TRIPLE"
if [ -n "$TARGET_TRIPLES" ] && [ $? != 0 ]; then
    echo "target_triple should be a value in this list: $TARGET_TRIPLES"
    exit -1
fi

# Load Python virtual environment
if [ "$GITLAB_CI" != "true" ]; then
    if [ ! -d ".venv" ]; then
        python3.12 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    fi

    echo "Activating python virtual environment"
    source .venv/bin/activate
    pip --retries 0 --timeout 5 install --upgrade pip

    # Install Buildstream if needed
    BST_VERSION=$(sed -n 's/^BuildStream==\([^ ]*\)$/\1/p' requirements_bst.txt)
    BST_PATH=".venv/bin/bst"
    if [ ! -x "${BST_PATH}" ] || [ $("${BST_PATH}" --version) != "${BST_VERSION}" ]; then
        pip install -r requirements_bst.txt
    fi
fi

export PATH="${PATH}:${HOME}/.local/bin"

command -v "pip3" &> /dev/null || { echo >&2 "'pip3' executable not found, please install python3-pip"; exit 1; }
command -v "flatpak" &> /dev/null || { echo >&2 "'flatpak' executable not found, please install Flatpak"; exit 1; }

source env.conf

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

GL_MOUNT_OPTS=""

export IS_TRUE_LINUX=$([[ "$(uname -s)" == "Linux" && -z "$(uname -a | grep -i 'microsoft')" ]] && echo 1 || echo 0)

if [[ "$IS_TRUE_LINUX" -eq 1 ]]; then # Enable GPU acceleration
    GL_BRANCH="26.08"
    GL_HOST_DIR="runtime/org.freedesktop.Platform.GL.default/x86_64/$GL_BRANCH/active/files"
    GL_HOST_DIR_USER="$HOME/.local/share/flatpak/$GL_HOST_DIR"
    GL_HOST_DIR_SYSTEM="/var/lib/flatpak/$GL_HOST_DIR"
    GL_EXTRA_HOST_DIR="runtime/org.freedesktop.Platform.GL.default/x86_64/$GL_BRANCH-extra/active/files"
    GL_EXTRA_HOST_DIR_USER="$HOME/.local/share/flatpak/$GL_EXTRA_HOST_DIR"
    GL_EXTRA_HOST_DIR_SYSTEM="/var/lib/flatpak/$GL_EXTRA_HOST_DIR"
    NVIDIA_VERSION_NAME=$(flatpak --gl-drivers|grep "nvidia") || true
    NVIDIA_HOST_DIR="runtime/org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME/x86_64/1.4/active/files"
    NVIDIA_HOST_DIR_USER="$HOME/.local/share/flatpak/$NVIDIA_HOST_DIR"
    NVIDIA_HOST_DIR_SYSTEM="/var/lib/flatpak/$NVIDIA_HOST_DIR"

    # The -extra and the NVIDIA runtimes are mounted inside the base one, and
    # bwrap makes the directory it mounts them over itself -- through the bind,
    # so it lands in the runtime on the host. That needs the runtime to be ours
    # to write to, which a user-wide flatpak install is and a system-wide one,
    # owned by root, is not: there bwrap fails outright and takes the sandbox
    # down with it. Mount them where that can work, and say so where it cannot.
    if [ -d "$GL_HOST_DIR_USER" ]; then
        GL_HOST_DIR_USED="$GL_HOST_DIR_USER"
        GL_EXTRA_HOST_DIR_USED="$GL_EXTRA_HOST_DIR_USER"
    elif [ -d "$GL_HOST_DIR_SYSTEM" ]; then
        GL_HOST_DIR_USED="$GL_HOST_DIR_SYSTEM"
        GL_EXTRA_HOST_DIR_USED="$GL_EXTRA_HOST_DIR_SYSTEM"
    fi

    # Either the mount point is already there, or bwrap has to be able to make it.
    gl_can_mount_into() { [ -d "$1/$2" ] || [ -w "$1" ]; }

    if [ -n "$GL_HOST_DIR_USED" ]; then
        GL_MOUNT_OPTS="--mount $GL_HOST_DIR_USED $GL_TARGET_DIR"
        if [ -d "$GL_EXTRA_HOST_DIR_USED" ] && gl_can_mount_into "$GL_HOST_DIR_USED" default; then
            GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $GL_EXTRA_HOST_DIR_USED $GL_TARGET_DIR/default"
        fi
    fi

    if [ -z "$NVIDIA_VERSION_NAME" ]; then
        echo "Please, install NVIDIA Drivers in order to have GPU acceleration."
    elif [ -d "$NVIDIA_HOST_DIR_USER" ]; then
        NVIDIA_HOST_DIR_USED="$NVIDIA_HOST_DIR_USER"
    elif [ -d "$NVIDIA_HOST_DIR_SYSTEM" ]; then
        NVIDIA_HOST_DIR_USED="$NVIDIA_HOST_DIR_SYSTEM"
    else
        echo "Please, install flatpaked NVIDIA Drivers in order to have GPU acceleration (flatpak install flathub org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME)"
    fi

    if [ -n "$NVIDIA_HOST_DIR_USED" ]; then
        if [ -z "$GL_HOST_DIR_USED" ] || gl_can_mount_into "$GL_HOST_DIR_USED" "$NVIDIA_VERSION_NAME"; then
            GL_MOUNT_OPTS="$GL_MOUNT_OPTS --mount $NVIDIA_HOST_DIR_USED $GL_TARGET_DIR/$NVIDIA_VERSION_NAME"
        else
            echo "Please, install the GL runtime as user in order to have NVIDIA GPU acceleration (flatpak install --user flathub org.freedesktop.Platform.GL.default//$GL_BRANCH): the system-wide one is read only, and the driver has to be mounted into it."
        fi
    fi
fi

#MOUNT_OPTS="$GL_MOUNT_OPTS --mount opencl_vendors /etc/opencl_vendors --mount /srv/tmp-squey /srv/tmp-squey"
# In CI, /srv/tmp-squey is a host volume shared by concurrent jobs: mount a
# per-slot subdirectory at the canonical sandbox path so jobs can never step
# on each other's files. Outside CI (no CI_CONCURRENT_ID) this is unchanged.
TMP_SQUEY_DIR="/srv/tmp-squey${CI_CONCURRENT_ID:+/slot-$CI_CONCURRENT_ID}"
mkdir -p "$TMP_SQUEY_DIR" 2>/dev/null || true
MOUNT_OPTS="$GL_MOUNT_OPTS --mount $TMP_SQUEY_DIR /srv/tmp-squey"
