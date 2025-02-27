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
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    fi

    echo "Activating python virtual environment"
    source .venv/bin/activate
    pip install --upgrade pip

    # Install Buildstream if needed
    BST_VERSION="2.3.0"
    BST_PATH=".venv/bin/bst"
    if [ ! -x "${BST_PATH}" ] || [ $("${BST_PATH}" --version) != "${BST_VERSION}" ]; then
        pip install BuildStream==${BST_VERSION}
        pip install -r requirements_bst.txt
        # Patch BuildStream to expose CAS socket in order to use recc from the build sandbox
        sed '135 i \            buildbox_command.append("--bind-mount={}:/tmp/casd.sock".format(casd._socket_path))\n' -i .venv/lib/python*/site-packages/buildstream/sandbox/_sandboxbuildboxrun.py
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
    GL_HOST_DIR="runtime/org.freedesktop.Platform.GL.default/x86_64/24.08/active/files"
    GL_HOST_DIR_USER="$HOME/.local/share/flatpak/$GL_HOST_DIR"
    GL_HOST_DIR_SYSTEM="/var/lib/flatpak/$GL_HOST_DIR"
    GL_EXTRA_HOST_DIR="runtime/org.freedesktop.Platform.GL.default/x86_64/24.08extra/active/files"
    GL_EXTRA_HOST_DIR_USER="$HOME/.local/share/flatpak/$GL_EXTRA_HOST_DIR"
    GL_EXTRA_HOST_DIR_SYSTEM="/var/lib/flatpak/$GL_EXTRA_HOST_DIR"
    NVIDIA_VERSION_NAME=$(flatpak --gl-drivers|grep "nvidia") || true
    NVIDIA_HOST_DIR="runtime/org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME/x86_64/1.4/active/files"
    NVIDIA_HOST_DIR_USER="$HOME/.local/share/flatpak/$NVIDIA_HOST_DIR"
    NVIDIA_HOST_DIR_SYSTEM="/var/lib/flatpak/$NVIDIA_HOST_DIR"

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
fi

#MOUNT_OPTS="$GL_MOUNT_OPTS --mount opencl_vendors /etc/opencl_vendors --mount /srv/tmp-squey /srv/tmp-squey"
MOUNT_OPTS="$GL_MOUNT_OPTS --mount /srv/tmp-squey /srv/tmp-squey"
