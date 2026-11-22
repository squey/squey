#!/bin/bash
#
# Strips the ELF files of an installed prefix, and keeps their debug information
# in <prefix>/lib/debug, where the debug extension of the Flatpak takes it from.
# Each stripped file names its debug file in a .gnu_debuglink section. Files of
# another format, PE or Mach-O, are left alone. squey.bst runs it on the
# installation of Squey.
#
# A file that names a debug file already was split before, and is only
# stripped: mold splits the binaries of Squey as it links them
# (--separate-debug-file in src/CMakeCompilers.txt), and the SDK the libraries
# install_linux_files.sh copies from it. Splitting such a file again would keep
# nothing but its symbol table, which no link would lead to, as objcopy refuses
# to add a second one.
#
# Usage: split_debug_info.sh <prefix>

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <prefix>" >&2
  exit 1
fi
DEBUG_DIR="$1/lib/debug"
export DEBUG_DIR

# The debug files are ELF files too, and the search would split them in turn if
# it came across them.
cd "$1" && find . -path ./lib/debug -prune -o -type f \
'(' -perm -111 -o -name '*.so*' \
    -o -name '*.cmxs' -o -name '*.node' ')' \
-exec bash -ec \
'read -n4 hdr <"$1" # check for elf header
 if [ "$hdr" != "$(printf \\x7fELF)" ]; then
     exit 0
 fi
 debugfile=
 if ! readelf --section-headers --wide "$1" | grep -qF " .gnu_debuglink "; then
     debugfile="$DEBUG_DIR/$1"
     mkdir -p "$(dirname "$debugfile")"
     objcopy --only-keep-debug --compress-debug-sections "$1" "$debugfile"
     chmod 644 "$debugfile"
 fi
 strip --remove-section=.comment --remove-section=.note --strip-unneeded "$1"
 if [ -n "$debugfile" ]; then
     objcopy --add-gnu-debuglink "$debugfile" "$1"
 fi' - {} ';'
