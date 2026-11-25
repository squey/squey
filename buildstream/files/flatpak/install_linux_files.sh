#!/bin/bash
#
# Installs what the Linux package needs beside what CMake installs: the files of
# the SDK pocl links the kernels it compiles for the processor against (see
# POCL_LINKER_DIR in portablecl.bst), which the runtime of the Flatpak does not
# carry, and the launchers of the Flatpak, which it adds to install_manifest.txt
# at the root of the prefix. squey.bst runs it once Squey is installed.
#
# Usage: install_linux_files.sh <prefix> <library triplet> <GCC triplet>
#
# <prefix>           the installed prefix to complete, /app or a staging copy
# <library triplet>  the directory of the libraries of the SDK under /usr/lib
# <GCC triplet>      the directory of the files of GCC under /usr/lib/gcc

set -e

if [ $# -ne 3 ]; then
  echo "Usage: $0 <prefix> <library triplet> <GCC triplet>" >&2
  exit 1
fi
PREFIX="$1"
LIBRARY_DIR="/usr/lib/$2"
GCC_DIR="/usr/lib/gcc/$3"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIBDIR="$PREFIX/lib"

install -D "$LIBRARY_DIR/crti.o" -t "$LIBDIR"
install -D "$LIBRARY_DIR/crtn.o" -t "$LIBDIR"
install -D "$LIBRARY_DIR/libgcc_s.so" -t "$LIBDIR"
install -D "$LIBRARY_DIR/libgcc_s.so.1" -t "$LIBDIR"
install -D "$LIBRARY_DIR/libc.so" -t "$LIBDIR"
sed -i "s|$LIBRARY_DIR/||g;s|/lib64/||" "$LIBDIR/libc.so"
install -D "$LIBRARY_DIR/libc_nonshared.a" -t "$LIBDIR"
cp "$LIBRARY_DIR/libm.so.6" "$LIBDIR/libm.so"
find "$GCC_DIR" -name libgcc.a -exec install -D {} -t "$LIBDIR" \;
find "$GCC_DIR" -name crtbeginS.o -exec install -D {} -t "$LIBDIR" \;
find "$GCC_DIR" -name crtendS.o -exec install -D {} -t "$LIBDIR" \;

# The launchers do not go through CMake: they add themselves to the files it
# installed, which squey-cleanup.bst keeps in bin/ (see squey.bst).
for launcher in run_cmd.sh squey_sh squey_launcher.sh; do
  cp "$SCRIPT_DIR/$launcher" "$PREFIX/bin"
  echo "$PREFIX/bin/$launcher" >> "$PREFIX/install_manifest.txt"
done
