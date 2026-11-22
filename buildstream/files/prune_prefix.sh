#!/bin/bash
#
# Prunes an installation prefix, in place, down to what the packages of Squey
# ship: no build or test byproducts, headers, static libraries, documentation,
# nor any executable but those Squey installs and those listed below.
# squey-cleanup.bst runs it before the Flatpak, the macOS bundle and the MSIX
# package are made of the prefix.
#
# Usage: prune_prefix.sh <prefix> <platform> <host> <python version>
#
# <platform>        linux, darwin or windows
# <host>            the target triplet, whose strip strips the Windows binaries
# <python version>  major.minor of the Python whose pip stays

set -e
set -x

if [ $# -ne 4 ]; then
  echo "Usage: $0 <prefix> <platform> <host> <python version>" >&2
  exit 1
fi
PREFIX="$1"
PLATFORM="$2"
HOST="$3"
PYTHON_VERSION="$4"

# What Squey installs in bin/ is kept, as install_manifest.txt, which CMake writes,
# lists it at the root of the prefix (see squey.bst). These are the tools of its
# dependencies to keep beside it: crashpad_handler captures the minidump out of
# process, for squey-crashreport to ask for consent and upload it.
common_bin_files="tshark lbzip2 funzip pigz zstd xz pip3 pip$PYTHON_VERSION crashpad_handler"
case "$PLATFORM" in
  linux) platform_bin_files="ld 7zzs" ;;
  darwin) platform_bin_files="lld ld64.lld 7zz" ;;
  windows) platform_bin_files='ld 7z python pocl "*.dll"' ;;
  *)
    echo "Unknown platform '$PLATFORM', expected linux, darwin or windows" >&2
    exit 1
    ;;
esac

rm -rf "$PREFIX/compilation_build"
rm -rf "$PREFIX/junit.xml"
rm -rf "$PREFIX/code_coverage_report"
# The installed testsuite is only needed by the macOS and Windows packaging,
# which turns $PREFIX/tests into testsuite.zip to be run on the target
# platform. Linux runs its tests from the build tree, so keep them out of
# the published package.
if [ "$PLATFORM" = "linux" ]; then
  rm -rf "$PREFIX/tests" "$PREFIX/lib/debug/tests"
  # The Flatpak keeps its data in its own cache unless the user chooses
  # otherwise (see PVNraw::default_working_directory()), not in the
  # directory of the host the configuration names.
  sed -i '/^nraw_tmp=/d' "$PREFIX/share/squey/squey/pvconfig.ini"
fi
rm -rf "$PREFIX/include"
rm -rf "$PREFIX/lib/cmake"
rm -rf "$PREFIX/sbin"
find "$PREFIX/lib" \( -name "*.a" -o -name "*.la" -o -name "*.prl" \) ! \( -name "libgcc.a" -o -name "libc_nonshared.a" \) -exec rm -rf {} \;
rm -rf "$PREFIX/man"
rm -rf "$PREFIX/examples"
rm -rf "$PREFIX"/doc*
rm -rf "$PREFIX/mkspecs"
cd "$PREFIX/share" && ls -I squey -I qt6 -I pocl -I doc -I applications -I icons -I app-info | xargs rm -rf
[ -d "$PREFIX/share/qt6" ] && cd "$PREFIX/share/qt6" && ls -I resources | xargs rm -rf
cd "$PREFIX/share/doc" && ls -I squey | xargs rm -rf

# The manifest names each file by the path it was installed at, under the root it
# was installed into, which the prefix may only be a copy of: only the name of
# what lies in a bin/ directory counts.
own_bin_files=$(sed -n 's|^.*/bin/\([^/]*\)$|\1|p' "$PREFIX/install_manifest.txt" | paste -sd ' ')
rm "$PREFIX/install_manifest.txt"
all_files="$own_bin_files $common_bin_files $platform_bin_files"
cmd=$(echo "$all_files" | sed 's/\([^ ]*\)/-I \1 -I \1.exe/g')
cmd="ls $cmd | xargs rm -rf"
cd "$PREFIX/bin" && eval "$cmd"

if [ "$PLATFORM" = "windows" ] ; then
  # The executables used to be stripped at link time by a '-s' in
  # CMakeCompilers.txt, which left no symbols to dump. They are stripped
  # here instead, after build.sh has sent the symbols to the crash server.
  # $PREFIX/tests is stripped too: make-msix-package.sh zips it whole into
  # testsuite.zip, and the DWARF of the test binaries was 96% of that archive.
  # PE keeps its debug information inside the binary, unlike Mach-O, which
  # leaves it in the object files -- hence no macOS counterpart to this.
  # A build that skips the testsuite has no such directory.
  strip_dirs="$PREFIX/lib $PREFIX/bin"
  [ -d "$PREFIX/tests" ] && strip_dirs="$strip_dirs $PREFIX/tests"
  # shellcheck disable=SC2086 # one directory per word
  find $strip_dirs \( -name "*.dll" -o -name "*.exe" \) | xargs "$HOST-strip" -s
fi
