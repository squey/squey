#!/bin/bash
#
# Gathers the host GPU userspace under one fixed path, so that the GPU
# devcontainer configuration can name it without knowing anything about this
# machine.
#
# Runs on the host, as the "initializeCommand" of .devcontainer/gpu, before the
# container exists. The flatpak runtime carrying the NVIDIA userspace is named
# after the driver version and can be installed for the user or system wide, so
# there is no path to commit: this builds a copy of it at a stable location and
# lets the container mount that instead. It is the same set of directories
# buildstream/.common.sh hands to "bst shell".
#
# Finding nothing is not an error. The container starts either way; it simply
# renders on the CPU, as the default configuration does.

set -e

FARM="${XDG_CACHE_HOME:-$HOME/.cache}/squey-devcontainer/gl"
GL_RUNTIME="runtime/org.freedesktop.Platform.GL.default/x86_64/25.08/active/files"
GL_EXTRA_RUNTIME="runtime/org.freedesktop.Platform.GL.default/x86_64/25.08-extra/active/files"

rm -rf "$FARM"
mkdir -p "$FARM"

for root in "$HOME/.local/share/flatpak" /var/lib/flatpak; do
    [ -d "$root/$GL_RUNTIME" ] || continue
    FLATPAK_ROOT="$root"
    break
done

if [ -z "$FLATPAK_ROOT" ]; then
    echo "No org.freedesktop.Platform.GL.default runtime; the container will render on the CPU." >&2
    echo "Install it with: flatpak install flathub org.freedesktop.Platform.GL.default//25.08" >&2
    exit 0
fi

# "cp -al" hardlinks every regular file instead of copying its content, so this
# costs no disk space and barely any time. It has to be a real copy of the
# tree, not a symlink to the flatpak install: the farm as a whole gets bind
# mounted into the container, and a symlink pointing outside of it, at the
# flatpak install's own absolute host path, would dangle in there -- that path
# was never mounted in, and the container has no way to resolve it.
cp -al "$FLATPAK_ROOT/$GL_RUNTIME/." "$FARM/"
[ -d "$FLATPAK_ROOT/$GL_EXTRA_RUNTIME" ] && cp -al "$FLATPAK_ROOT/$GL_EXTRA_RUNTIME" "$FARM/default"

DRIVER="$(flatpak --gl-drivers 2>/dev/null | grep '^nvidia' | head -1)"
if [ -z "$DRIVER" ]; then
    echo "No NVIDIA flatpak driver in use; the container will render on the CPU." >&2
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
        # .devcontainer/gpu/devcontainer.json is static text and cannot glob for
        # whichever directory this run created, so LD_LIBRARY_PATH there is
        # written against this fixed alias instead.
        ln -sfn "$DRIVER" "$FARM/nvidia"
        echo "GPU userspace ready: $DRIVER"
        exit 0
    fi
done

echo "Driver $DRIVER is in use but its flatpak runtime is missing; the container will render on the CPU." >&2
echo "Install it with: flatpak install flathub org.freedesktop.Platform.GL.$DRIVER" >&2
