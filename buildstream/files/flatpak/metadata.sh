# shellcheck shell=bash disable=SC2034
#
# The metadata of the Flatpak. buildstream/scripts/export_flatpak.sh writes it
# into every build of the Flatpak, .gitlab-ci/flathub_publish.sh into the
# manifest Flathub builds from, and install_desktop_integration.sh gives the
# command to the desktop entry, which is what the desktop runs. The branch of the
# runtime is the one buildstream/app.yml names, see
# buildstream/scripts/runtime_branch.sh.

FLATPAK_APP_ID=org.squey.Squey
FLATPAK_RUNTIME=org.freedesktop.Platform
FLATPAK_SDK=org.freedesktop.Sdk

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
