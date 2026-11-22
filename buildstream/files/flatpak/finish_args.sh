# shellcheck shell=bash disable=SC2034
#
# What "flatpak build-finish" adds to the metadata of the Flatpak: the command it
# runs, and the permissions it is granted. buildstream/scripts/export_flatpak.sh
# finishes the Flatpak builds with them, and .gitlab-ci/flathub_publish.sh writes
# them into the manifest Flathub builds from.

FLATPAK_COMMAND=squey_launcher.sh

FLATPAK_PERMISSIONS=(
  --socket=wayland
  --socket=fallback-x11
  --share=network
  --share=ipc
  # The portals hand over the files the user picks, and forward those opened from
  # the desktop. Without the user, Squey reads and writes ~/.squey, and reads the
  # profiles of Wireshark.
  --filesystem=~/.squey:create
  --filesystem=xdg-config/wireshark:ro
  --device=dri
  --allow=devel
)
