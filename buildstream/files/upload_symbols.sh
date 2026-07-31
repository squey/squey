#!/bin/bash
#
# Generate Breakpad symbol files for the binaries of a build and upload them to
# the crash server. Runs on the build host for the three targets: the symbol
# dumper is picked from the binary format rather than from the host.
#
# usage: upload_symbols.sh <target-platform> <binaries-dir> <version> [url]
#
# Symbols have to be dumped before squey-cleanup.bst strips the Windows DLLs,
# otherwise there is nothing left to read.

set -eu

TARGET_PLATFORM="$1"
BINARIES_DIR="$2"
VERSION="$3"
UPLOAD_URL="${4:-https://squey.bugsplat.com/post/bp/symbol/breakpadsymbols.php?appName=Squey&appVer=$VERSION}"

SYM_DIR="$(mktemp -d)"
trap 'rm -rf "$SYM_DIR"' EXIT

# Executables carry no extension outside of Windows, so they are matched on the
# executable bit: squey, squey-crashreport and crashpad_handler all need symbols.
# Whatever is not a binary of the right format is reported as skipped below.
case "$TARGET_PLATFORM" in
  linux)   DUMP_SYMS=dump_syms;       BINARY_PATTERN=( -name "*.so*" -o -perm -111 ) ;;
  windows) DUMP_SYMS=dump_syms_dwarf; BINARY_PATTERN=( -name "*.dll" -o -name "*.exe" ) ;;
  darwin)  DUMP_SYMS=dump_syms_mac;   BINARY_PATTERN=( -name "*.dylib" -o -perm -111 ) ;;
  *) echo "unsupported target platform: $TARGET_PLATFORM" >&2; exit 1 ;;
esac

if ! command -v "$DUMP_SYMS" > /dev/null; then
  echo "$DUMP_SYMS not found, is breakpad-tools.bst part of the build dependencies?" >&2
  exit 1
fi

dumped=0
failed=0
while IFS= read -r binary; do
  sym_file="$SYM_DIR/$(basename "$binary").sym"
  if "$DUMP_SYMS" "$binary" > "$sym_file" 2> /dev/null && [ -s "$sym_file" ]; then
    # Keep the paths of the source tree out of the symbol files.
    sed 's|/buildstream/squey/squey.bst/||' -i "$sym_file"
    dumped=$((dumped + 1))
  else
    # A binary without debug information is not an error: third-party libraries
    # are shipped stripped.
    rm -f "$sym_file"
    failed=$((failed + 1))
  fi
# Matching on the executable bit also catches the test binaries and the probes
# CMake leaves behind, whose symbols are of no use to anyone looking at a crash.
done < <(find "$BINARIES_DIR" -type f ! -name "SQUEY_TEST_*" ! -name "CMake*" \
              \( "${BINARY_PATTERN[@]}" \))

echo "symbols: $dumped file(s) dumped, $failed skipped"

# sym_upload is not built by the fork (its Makefile.am lists its sources twice,
# which breaks the link), and the v1 protocol it speaks is a plain multipart
# POST whose fields all come from the MODULE line:
#   MODULE <os> <cpu> <debug_identifier> <debug_file>
uploaded=0
for sym_file in "$SYM_DIR"/*.sym; do
  [ -e "$sym_file" ] || continue

  read -r _ sym_os sym_cpu sym_id sym_name < "$sym_file"
  # A failed upload must not abort the build, hence the fallback under 'set -e'.
  http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "os=$sym_os" \
    -F "cpu=$sym_cpu" \
    -F "debug_identifier=${sym_id//-/}" \
    -F "debug_file=$sym_name" \
    -F "code_file=$sym_name" \
    -F "version=$VERSION" \
    -F "symbol_file=@$sym_file" \
    "$UPLOAD_URL") || http_code="000"

  if [ "$http_code" = "200" ]; then
    uploaded=$((uploaded + 1))
  else
    echo "failed to upload $(basename "$sym_file"): HTTP $http_code" >&2
  fi
done

echo "symbols: $uploaded file(s) uploaded to the crash server"
