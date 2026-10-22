#!/bin/bash

# Flatpak passes the host's QT_PLUGIN_PATH on, and Qt searches it first. The
# plugins of an older Qt it finds there pass the version check, then fail to
# load against this one, and Squey aborts with no platform plugin left.
unset QT_PLUGIN_PATH

run_cmd.sh squey "$@"
