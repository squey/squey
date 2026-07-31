#!/bin/bash
#
# Generate Breakpad symbol files for the binaries of a build and upload them to
# the crash server. Runs on the build host for the three targets: the symbol
# dumper is picked from the binary format rather than from the host.
#
# usage: upload_symbols.sh [--check-only] <target-platform> <binaries-dir> <version> [url]
#
# The symbols are checked before being sent, and nothing is sent if the check
# fails: symbols the crash server cannot match against a minidump are worse than
# none, as they make the reports look symbolized while every frame stays blank.
# --check-only runs that check on its own, for a build that has nothing to
# publish.
#
# Symbols have to be dumped before squey-cleanup.bst strips the Windows DLLs,
# otherwise there is nothing left to read.

set -eu

CHECK_ONLY=false
if [ "${1:-}" = "--check-only" ]; then
  CHECK_ONLY=true
  shift
fi

TARGET_PLATFORM="$1"
BINARIES_DIR="$2"
# A check needs no version, nothing being published.
VERSION="${3:-}"
UPLOAD_URL="${4:-https://squey.bugsplat.com/post/bp/symbol/breakpadsymbols.php?appName=Squey&appVer=$VERSION}"

SYM_DIR="$(mktemp -d)"
trap 'rm -rf "$SYM_DIR"' EXIT

# Executables carry no extension outside of Windows, so they are matched on the
# executable bit: squey, squey-crashreport and crashpad_handler all need symbols.
# Whatever is not a binary of the right format is reported as skipped below.
case "$TARGET_PLATFORM" in
  # A .dbg of an executable carries neither the .so of a library nor the
  # executable bit of the binary it was split from, so it needs a rule of its own.
  linux)   DUMP_SYMS=dump_syms;       BINARY_PATTERN=( -name "*.so*" -o -name "*.dbg" -o -perm -111 ) ;;
  windows) DUMP_SYMS=dump_syms_dwarf; BINARY_PATTERN=( -name "*.dll" -o -name "*.exe" ) ;;
  darwin)  DUMP_SYMS=dump_syms_mac;   BINARY_PATTERN=( -name "*.dylib" -o -perm -111 ) ;;
  *) echo "unsupported target platform: $TARGET_PLATFORM" >&2; exit 1 ;;
esac

# Two of the three targets keep the debug information apart from the binary it
# was built from: mold is given --separate-debug-file on Linux and writes a .dbg,
# while macOS gets a .dSYM bundle holding a file named after the binary. Both
# describe the very module the stripped binary stands for, so whichever is dumped
# last wins the symbol file, and a stripped binary winning means publishing a
# module without a single function under the name the minidump asks for.
debug_companion() {
  local binary="$1"
  local dsym="$binary.dSYM/Contents/Resources/DWARF/$(basename "$binary")"
  if [ -e "$binary.dbg" ]; then
    echo "$binary.dbg"
  elif [ -e "$dsym" ]; then
    echo "$dsym"
  fi
}

select_binaries() {
  find "$BINARIES_DIR" -type f ! -name "SQUEY_TEST_*" ! -name "PVCOP_TEST_*" ! -name "CMake*" \
       \( "${BINARY_PATTERN[@]}" \) | while IFS= read -r binary; do
    case "$binary" in
      # Reached through the binary it belongs to, never on its own.
      */Contents/Resources/DWARF/*) ;;
      # A debug file whose binary is gone is all that is left to dump.
      *.dbg) [ -e "${binary%.dbg}" ] || echo "$binary" ;;
      *) companion="$(debug_companion "$binary")"; echo "${companion:-$binary}" ;;
    esac
  done
}

if ! command -v "$DUMP_SYMS" > /dev/null; then
  echo "$DUMP_SYMS not found, is breakpad-tools.bst part of the build dependencies?" >&2
  exit 1
fi

dumped=0
failed=0
while IFS= read -r binary; do
  sym_file="$SYM_DIR/$(basename "${binary%.dbg}").sym"
  if "$DUMP_SYMS" "$binary" > "$sym_file" 2> /dev/null && [ -s "$sym_file" ]; then
    # Keep the paths of the source tree out of the symbol files.
    sed 's|/buildstream/squey/squey.bst/||' -i "$sym_file"
    # The MODULE line of a .dbg names the .dbg itself, while the minidump names
    # the binary that was loaded: without this the two never meet.
    sed '1s|\.dbg$||' -i "$sym_file"
    dumped=$((dumped + 1))
  else
    # A binary without debug information is not an error: third-party libraries
    # are shipped stripped.
    rm -f "$sym_file"
    failed=$((failed + 1))
  fi
done < <(select_binaries)

echo "symbols: $dumped file(s) dumped, $failed skipped"

# 'libpvkernel.so.1', 'libpvkernel.dll' and 'libpvkernel.1.dylib' are the same
# module under three targets, so the check is written against what is left once
# the decoration of the platform is taken off.
module_stem() {
  local name
  name="$(basename "$1" .sym)"
  name="${name%.exe}"; name="${name%.dll}"; name="${name%.dylib}"
  name="${name%%.so*}"; name="${name%%.[0-9]*}"
  echo "${name#lib}"
}

# Nobody can read a crash report of squey that names no function of squey. These
# are the modules a backtrace is made of, and their symbols went out empty for a
# year without anyone noticing, hence the check.
REQUIRED_MODULES=( squey squey-crashreport pvkernel pvguiqt )

errors=0

for required in "${REQUIRED_MODULES[@]}"; do
  found=false
  for sym_file in "$SYM_DIR"/*.sym; do
    [ -e "$sym_file" ] || continue
    [ "$(module_stem "$sym_file")" = "$required" ] || continue
    # A symbol file is written even when nothing could be read from the debug
    # information, and holds the PUBLIC entries of the symbol table alone: it
    # names the module of a frame but never its function.
    if grep -q "^FUNC " "$sym_file"; then
      found=true
      break
    fi
    echo "check: $(basename "$sym_file") holds no function" >&2
  done
  if [ "$found" = false ]; then
    echo "check: no usable symbols for $required" >&2
    errors=$((errors + 1))
  fi
done

# The crash server looks the symbols up under the identifier the minidump gives,
# which names the binary that was loaded, not the debug file the symbols were
# read from. Both derive it from the build id, so a binary lacking one is given
# an identifier of its own and its symbols are never found again.
while IFS= read -r debug_file; do
  binary="${debug_file%.dbg}"
  [ -e "$binary" ] || continue
  binary_id="$("$DUMP_SYMS" -i "$binary" 2> /dev/null | head -1 | cut -d' ' -f4)"
  debug_id="$("$DUMP_SYMS" -i "$debug_file" 2> /dev/null | head -1 | cut -d' ' -f4)"
  if [ -z "$binary_id" ] || [ "$binary_id" != "$debug_id" ]; then
    echo "check: $(basename "$binary") is identified as '${binary_id:-none}' but its" \
         "symbols as '${debug_id:-none}', the crash server would never match them" >&2
    errors=$((errors + 1))
  fi
done < <(find "$BINARIES_DIR" -type f -name "*.dbg" ! -name "SQUEY_TEST_*" ! -name "PVCOP_TEST_*")

if [ "$errors" -gt 0 ]; then
  echo "symbols: $errors check(s) failed, nothing uploaded" >&2
  exit 1
fi

echo "symbols: checks passed"

if [ "$CHECK_ONLY" = true ]; then
  exit 0
fi

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
