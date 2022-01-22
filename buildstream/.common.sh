#!/bin/bash

export PATH="${PATH}:${HOME}/.local/bin"

command -v "pip3" &> /dev/null || { echo >&2 "'pip3' executable not found, please install python3-pip"; exit 1; }
command -v "flatpak" &> /dev/null || { echo >&2 "'flatpak' executable not found, please install Flatpak"; exit 1; }

source env.conf

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -z "$WORKSPACE_PREFIX" ]; then
    WORKSPACE_PREFIX="$DIR"
fi

NVIDIA_VERSION_NAME=$(flatpak --gl-drivers|grep "nvidia") || true
GL_HOST_DIR="$HOME/.local/share/flatpak/runtime/org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME/x86_64/1.4/active/files"
GL_MOUNT_OPTS=""
if [ -d "$GL_HOST_DIR" ]; then
    GL_MOUNT_OPTS="--mount $GL_HOST_DIR $GL_TARGET_DIR"
else
    if [ -z "$NVIDIA_VERSION_NAME" ]; then
        echo "Please, install NVIDIA Drivers in order to have GPU acceleration."
    else
        echo "Please, install flatpaked NVIDIA Drivers in order to have GPU acceleration (flatpak install flathub org.freedesktop.Platform.GL.$NVIDIA_VERSION_NAME)"
    fi
fi
MOUNT_OPTS="$GL_MOUNT_OPTS --mount opencl_vendors /etc/opencl_vendors --mount /srv/tmp-inspector /srv/tmp-inspector"

# Install Buildstream and bst-external plugins if needed
command -v "bst" &> /dev/null || { pip install --user BuildStream==1.6.3; }
python3 -c "import bst_external" &> /dev/null || pip install --user -e "$DIR/plugins/bst-external"

function check_bindfs()
{
    command -v "bindfs" &> /dev/null || { echo >&2 "'bindfs' executable not found, please install bindfs (then log out and log back in)"; exit 1; }
}

# Use workspace to have persistence over CCACHE_DIR
function open_workspace()
{
    WORKSPACE_NAME="$1"
    WORKSPACE_PATH="$WORKSPACE_PREFIX/$WORKSPACE_NAME"

    CURRENT_WORKSPACE=`bst workspace list | grep "directory:" | awk -F ': ' '{print $2}'`
    if [ "$CURRENT_WORKSPACE" != "$WORKSPACE_PATH" ]; then
        if [ "$WORKSPACE_NAME" == "workspace_build" ]; then
            check_bindfs
            mkdir -p "$DIR/../release_build" "$DIR/../debug_build"
            bindfs --no-allow-other -o nonempty "$DIR/empty/" "$DIR/../release_build"
            bindfs --no-allow-other -o nonempty "$DIR/empty/" "$DIR/../debug_build"
        elif [ "$WORKSPACE_NAME" == "workspace_dev" ] && [ -d "$DIR/workspace_build" ]; then
            check_bindfs
            bindfs --no-allow-other -o nonempty "$DIR/empty/" "$DIR/workspace_build"
        fi
    
        bst workspace close inendi-inspector.bst || true
        bst fetch freedesktop-sdk.bst
        if [ ! -d "$WORKSPACE_PATH" ]; then
            bst workspace open inendi-inspector.bst $WORKSPACE_PATH
        else
            bst workspace open --no-checkout inendi-inspector.bst $WORKSPACE_PATH
        fi
        
        if [ "$WORKSPACE_NAME" == "workspace_build" ]; then
            fusermount -u "$DIR/../release_build"
            fusermount -u "$DIR/../debug_build"
        elif [ "$WORKSPACE_NAME" == "workspace_dev" ] && [ -d "$DIR/workspace_build" ]; then
            fusermount -u "$DIR/workspace_build"
        fi
    fi
}
