#!/bin/bash
#
# Gathers what the GPU of this machine needs under fixed paths, its devices and
# its userspace, and the Wayland socket with them, so that
# .devcontainer/devcontainer.json can name them without knowing anything about
# this machine.
#
# Runs on the host, as the "initializeCommand", before the container is created
# or started. The flatpak runtime carrying the NVIDIA userspace is named after
# the driver version and can be installed for the user or system wide, so there
# is no path to commit: this builds a copy of it at a stable location and lets
# the container mount that instead. It is the same set of directories
# buildstream/.common.sh hands to "bst shell".
#
# Finding nothing is not an error. The container starts either way; it simply
# renders on the CPU, and shows no window without a Wayland session.

set -e

# The paths .devcontainer/devcontainer.json names, which cannot follow
# XDG_CACHE_HOME
DEVICES="$HOME/.cache/squey-devcontainer/dev"
WAYLAND_SOCKET="$HOME/.cache/squey-devcontainer/wayland"
FARM="$HOME/.cache/squey-devcontainer/gl"
# The GL runtime of the freedesktop SDK the image is made of, which
# buildstream/.common.sh names for the sandbox
GL_BRANCH="$(sed -n 's/^[[:space:]]*GL_BRANCH="\(.*\)"$/\1/p' "$(dirname "${BASH_SOURCE[0]}")/../buildstream/.common.sh")"
[ -n "$GL_BRANCH" ] || { echo "buildstream/.common.sh names no GL_BRANCH." >&2; exit 1; }
GL_RUNTIME="runtime/org.freedesktop.Platform.GL.default/x86_64/$GL_BRANCH/active/files"
GL_EXTRA_RUNTIME="runtime/org.freedesktop.Platform.GL.default/x86_64/$GL_BRANCH-extra/active/files"

# A device the container is created with has to exist, or the container refuses
# to start, and devcontainer.json cannot say "if present". It names these links
# instead, which always exist: each points at the device when this machine has
# it, and at /dev/null otherwise, a harmless stand-in for the container to get.
# The container runtime resolves them when it creates the container, after this
# script.
mkdir -p "$DEVICES"
for device in /dev/dri /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
    target="$device"
    [ -e "$target" ] || target=/dev/null
    ln -sfn "$target" "$DEVICES/${device##*/}"
done
# The same goes for the source of a bind mount, and so for the Wayland socket,
# which a host without a Wayland session lacks. WAYLAND_DISPLAY names it,
# relative to XDG_RUNTIME_DIR unless it is a path.
wayland="${WAYLAND_DISPLAY:-wayland-0}"
[[ "$wayland" == /* ]] || wayland="$XDG_RUNTIME_DIR/$wayland"
[ -S "$wayland" ] || wayland=/dev/null
ln -sfn "$wayland" "$WAYLAND_SOCKET"

# Emptied rather than removed: a running container keeps this very directory
# mounted and would be left holding a deleted one, and this script runs again
# whenever the CLI opens a container, running or not, as well as whenever
# another checkout creates one.
mkdir -p "$FARM"
find "$FARM" -mindepth 1 -delete

for root in "$HOME/.local/share/flatpak" /var/lib/flatpak; do
    [ -d "$root/$GL_RUNTIME" ] || continue
    FLATPAK_ROOT="$root"
    break
done

if [ -z "$FLATPAK_ROOT" ]; then
    echo "No org.freedesktop.Platform.GL.default runtime; the container will render on the CPU." >&2
    echo "Install it with: flatpak install flathub org.freedesktop.Platform.GL.default//$GL_BRANCH" >&2
    exit 0
fi

# "cp -al" hardlinks every regular file instead of copying its content, so this
# costs no disk space and barely any time. It has to be a real copy of the
# tree, not a symlink to the flatpak install: the farm as a whole gets bind
# mounted into the container, and a symlink pointing outside of it, at the
# flatpak install's own absolute host path, would dangle in there -- that path
# was never mounted in, and the container has no way to resolve it.
cp -al "$FLATPAK_ROOT/$GL_RUNTIME/." "$FARM/"
# Into the empty "default" directory GL.default ships for it, where the sandbox
# mounts it too: the trailing "/." is explained below.
[ -d "$FLATPAK_ROOT/$GL_EXTRA_RUNTIME" ] && cp -al "$FLATPAK_ROOT/$GL_EXTRA_RUNTIME/." "$FARM/default"

DRIVER="$(flatpak --gl-drivers 2>/dev/null | grep '^nvidia' | head -1)"
if [ -z "$DRIVER" ]; then
    echo "No NVIDIA flatpak driver in use; the container will draw on an AMD or Intel GPU if there is one, on the CPU otherwise." >&2
    exit 0
fi

NVIDIA_RUNTIME="runtime/org.freedesktop.Platform.GL.$DRIVER/x86_64/1.4/active/files"
for root in "$HOME/.local/share/flatpak" /var/lib/flatpak; do
    if [ -d "$root/$NVIDIA_RUNTIME" ]; then
        # The trailing "/." matters: GL.default ships an empty "$DRIVER" directory
        # of its own, a mount point meant for a flatpak extension to be layered
        # onto, which the copy above already recreated. Copying into an already
        # existing directory nests under its basename ("files") instead of
        # flattening into it; forcing the source to be read as "its contents",
        # not "itself", avoids that regardless of whether the destination
        # existed going in.
        mkdir -p "$FARM/$DRIVER"
        cp -al "$root/$NVIDIA_RUNTIME/." "$FARM/$DRIVER"
        # A second, driver-version-independent name, relative and staying inside
        # the farm so it survives the bind mount same as everything else here:
        # .devcontainer/devcontainer.json is static text and cannot glob for
        # whichever directory this run created, so LD_LIBRARY_PATH there is
        # written against this fixed alias instead.
        ln -sfn "$DRIVER" "$FARM/nvidia"
        echo "GPU userspace ready: $DRIVER"
        exit 0
    fi
done

echo "Driver $DRIVER is in use but its flatpak runtime is missing; the container will not draw on that GPU." >&2
echo "Install it with: flatpak install flathub org.freedesktop.Platform.GL.$DRIVER" >&2
