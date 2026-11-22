#!/bin/bash
#
# Strips the ELF files of an installed prefix, and keeps their debug information
# in <prefix>/lib/debug, where the debug extension of the Flatpak takes it from.
# Each stripped file names its debug file in a .gnu_debuglink section. Files of
# another format, PE or Mach-O, are left alone. squey.bst runs it on the
# installation of Squey.
#
# Usage: split_debug_info.sh <prefix>

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <prefix>" >&2
  exit 1
fi
DEBUG_DIR="$1/lib/debug"
export DEBUG_DIR

cd "$1" && find . -type f \
'(' -perm -111 -o -name '*.so*' \
    -o -name '*.cmxs' -o -name '*.node' ')' \
-exec bash -ec \
'read -n4 hdr <"$1" # check for elf header
 if [ "$hdr" != "$(printf \\x7fELF)" ]; then
     exit 0
 fi
 debugfile="$DEBUG_DIR/$1"
 mkdir -p "$(dirname "$debugfile")"
 objcopy --only-keep-debug --compress-debug-sections "$1" "$debugfile"
 chmod 644 "$debugfile"
 strip --remove-section=.comment --remove-section=.note --strip-unneeded "$1"
 objcopy --add-gnu-debuglink "$debugfile" "$1"' - {} ';'
