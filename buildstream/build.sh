#!/bin/bash

TMP_ARTIFACT_DIR="$(mktemp -d)"

function cleanup {
  rm -rf -- "${TMP_ARTIFACT_DIR}"
  # The BuildStream cache location honors XDG_CACHE_HOME, which CI sets per
  # runner slot: only ever clean the cache of our own slot, never the shared
  # one (a job finishing used to wipe directories still in use by concurrent
  # jobs). Falls back to ~/.cache for local builds.
  local bst_cache="${XDG_CACHE_HOME:-$HOME/.cache}/buildstream"
  rm -rf "${bst_cache}/artifacts/extract/squey/squey"
  rm -rf "${bst_cache}/build"
  # Same per-slot isolation as MOUNT_OPTS in .common.sh ("tomjon" is the
  # default sandbox user of BuildStream, see its projectconfig.yaml).
  rm -rf "/srv/tmp-squey${CI_CONCURRENT_ID:+/slot-$CI_CONCURRENT_ID}/tomjon"/*
}

trap cleanup EXIT SIGKILL SIGQUIT SIGSEGV SIGABRT

usage() {
echo "Usage: $0"
  echo "--target_triple=<cross-compilation_target_triple>"
  echo "--branch=<branch_name_or_tag_name>"
  echo "--code-coverage=<true/false>"
  echo "--cxx_compiler=<g++/clang++>"
  echo "--disable-testsuite=<true/false>"
  echo "--export=<true/false>"
  echo "--export-dir=<repository_path>"
  echo "--macos-sdk-dir=<macos_sdk_dir>"
  echo "--gpg-private-key-path=<key>"
  echo "--gpg-sign-key=<key>"
  echo "--push-artifacts=<true/false>"
  echo "--upload-debug-symbols=<true/false>"
  echo "--user-target=<USER_TARGET>" 1>&2; exit 1;
}

# Set default options
TARGET_TRIPLE="x86_64-linux-gnu"
BRANCH_NAME=main
TAG_NAME=
BUILD_TYPE=RelWithDebInfo
CXX_COMPILER=clang++
USER_TARGET=developer
USER_TARGET_SPECIFIED=false
EXPORT_BUILD=false
EXPORT_DIR="export"
MACOS_SDK_DIR=""
TESTSUITE_DISABLED=false
GPG_PRIVATE_KEY_PATH=
GPG_SIGN_KEY=
CODE_COVERAGE_ENABLED=false
UPLOAD_DEBUG_SYMBOLS=false
PUSH_ARTIFACTS=false

# Override default options with user provided options
OPTS=`getopt -o h:r:m:b:t:d:g:k:e:p,l,u,a,t,s --long help,target_triple:,export:,export-dir:,macos-sdk-dir:,gpg-private-key-path:,gpg-sign-key:,branch:,build-type:,cxx-compiler:,user-target:,disable-testsuite:,code-coverage:,upload-debug-symbols:,push-artifacts: -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ; exit 0 ;;
    -t | --target_triple ) TARGET_TRIPLE="$2"; shift 2 ;;
    -b | --branch ) BRANCH_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -p | --cxx-compiler ) CXX_COMPILER="$2"; shift 2 ;;
    -m | --user-target ) USER_TARGET_SPECIFIED=true; USER_TARGET="$2"; shift 2 ;;
    -d | --disable-testsuite ) TESTSUITE_DISABLED="$2"; shift 2 ;;
    -e | --export ) EXPORT_BUILD="$2"; shift 2 ;;
    -r | --export-dir ) EXPORT_DIR="$2"; shift 2 ;;
    -s | --macos-sdk-dir ) MACOS_SDK_DIR="$2"; shift 2 ;;
    -g | --gpg-private-key-path ) GPG_PRIVATE_KEY_PATH="$2"; shift 2 ;;
    -k | --gpg-sign-key ) GPG_SIGN_KEY="$2"; shift 2 ;;
    -l | --code-coverage ) CODE_COVERAGE_ENABLED="$2"; shift 2 ;;
    -u | --upload-debug-symbols ) UPLOAD_DEBUG_SYMBOLS="$2"; shift 2 ;;
    -a | --push-artifacts ) PUSH_ARTIFACTS="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

source .common.sh

set -e
set -x

./generate_appstream_metadata.sh

if [ -n "$MACOS_SDK_DIR" ]; then
  MACOS_SDK_LOCAL_DIR="files/macos_sdk"
  mkdir -p "$MACOS_SDK_LOCAL_DIR"
  cp "$MACOS_SDK_DIR"/* "$MACOS_SDK_LOCAL_DIR"
fi

# Build Squey
BUILD_OPTIONS="--option target_triple $TARGET_TRIPLE --option cxx_compiler $CXX_COMPILER --error-lines 10000 "
if [ $USER_TARGET_SPECIFIED = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option user_target $USER_TARGET"
fi
if  [ "$UPLOAD_DEBUG_SYMBOLS" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option keep_build_dir True"
fi
if  [ "$TESTSUITE_DISABLED" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option disable_testsuite True"
fi
if  [ "$GITLAB_CI" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option quiet_compilation True"
fi
if  [ "$CODE_COVERAGE_ENABLED" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option code_coverage True"
  pushd .
  cd ../src/libpvcop/tests/files
  git submodule update --checkout
  popd
fi

# Dump the symbols of the freshly built binaries and send them to the crash
# server, so that the minidumps Crashpad captures can be symbolized. The build
# directory is only kept around when UPLOAD_DEBUG_SYMBOLS is set, and the
# symbols have to be read before squey-cleanup.bst strips the binaries.
# The script is fed through stdin as it lives in the source tree, which is not
# staged in a non-build shell.
upload_debug_symbols() {
  local target_platform="$1"
  if [ "$UPLOAD_DEBUG_SYMBOLS" != true ]; then
    return 0
  fi
  local version="$(cat ../VERSION.txt)"
  bst $BUILD_OPTIONS shell $MOUNT_OPTS squey.bst -- \
    bash -s -- "$target_platform" /compilation_build "$version" \
    < files/upload_symbols.sh
}

if [ "$EXPORT_BUILD" = false ]; then
  bst $BUILD_OPTIONS build --retry-failed squey.bst
elif [ "$TARGET_TRIPLE" == "x86_64-linux-gnu" ]; then # Generate Linux flatpak repository

  if [[ ! -z "$GPG_PRIVATE_KEY_PATH" ]]; then
    # Import GPG private key
    gpg --import --no-tty --batch --yes $GPG_PRIVATE_KEY_PATH
  fi

  # Export flatpak Release image
  bst $BUILD_OPTIONS build flatpak/org.squey.Squey.bst
  upload_debug_symbols linux
  bst $BUILD_OPTIONS artifact checkout flatpak/org.squey.Squey.bst --directory "$TMP_ARTIFACT_DIR/flatpak_files"
  mkdir -p "$EXPORT_DIR" &> /dev/null || true
  if [[ ! -z "$GPG_SIGN_KEY" ]]; then
    flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $EXPORT_DIR "$TMP_ARTIFACT_DIR/flatpak_files" $BRANCH_NAME
  else
    flatpak build-export --files=files $EXPORT_DIR "$TMP_ARTIFACT_DIR/flatpak_files" $BRANCH_NAME
  fi

  ## Export flatpak Debug image
  #rm -rf $DIR/build
  #bst $BUILD_OPTIONS build flatpak/org.squey.Squey.Debug.bst
  #bst $BUILD_OPTIONS checkout flatpak/org.squey.Squey.Debug.bst "$DIR/build"
  #if [[ ! -z "$GPG_SIGN_KEY" ]]; then
  #  flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $EXPORT_DIR $DIR/build $BRANCH_NAME
  #else
  #  flatpak build-export --files=files $EXPORT_DIR $DIR/build $BRANCH_NAME
  #fi
elif [ "$TARGET_TRIPLE" == "x86_64-apple-darwin" ] || [ "$TARGET_TRIPLE" == "aarch64-apple-darwin" ]; then # Generate MacOS app bundle
  bst $BUILD_OPTIONS build macos_bundle/dmg-image.bst
  upload_debug_symbols darwin
  rm -rf "$EXPORT_DIR/$TARGET_TRIPLE"
  mkdir -p "$EXPORT_DIR/$TARGET_TRIPLE"
  bst $BUILD_OPTIONS shell -b --mount "$EXPORT_DIR/$TARGET_TRIPLE" /output macos_bundle/dmg-image.bst bash "buildstream/files/macos_bundle/make-dmg-image.sh"
elif [ "$TARGET_TRIPLE" == "x86_64-w64-mingw32" ]; then # Generate Windows MSIX package
  bst $BUILD_OPTIONS build msix_package/msix-package.bst
  upload_debug_symbols windows
  rm -rf "$EXPORT_DIR/$TARGET_TRIPLE"
  mkdir -p "$EXPORT_DIR/$TARGET_TRIPLE"
  bst $BUILD_OPTIONS shell -b --mount "$EXPORT_DIR/$TARGET_TRIPLE" /output msix_package/msix-package.bst bash "buildstream/files/msix_package/make-msix-package.sh"
fi

# BuildStream only records the output of the build commands in the per-element build
# log, and only prints it on failure, so the testsuite results have to be surfaced
# explicitly. The code coverage case is skipped as it already dumps the whole log.
if [ "$CODE_COVERAGE_ENABLED" = false ]; then
  bst $BUILD_OPTIONS artifact log squey.bst | sed -n '/Test project/,/Total Test time/p' || true
fi

# Hand over to the shared artifact cache (ARTIFACT_CACHE_URL, the pool the CI
# runner hosts) what a later build can reuse. Its remotes are declared pull-only
# in the runner configuration, so a build never uploads anything by itself: this
# is the only place that pushes, and it leaves squey.bst out. That one is the
# largest artifact of the build and the next commit makes it obsolete, whereas
# its dependencies are worth the room they take, the freedesktop-sdk included:
# once upstream retention drops them, the alternative is to build them from
# source again. Sources stay out of the pool altogether, they are quick to fetch
# and stable enough to not be worth the room.
# A failure here costs a redundant rebuild later on, never a job.
if [ "$PUSH_ARTIFACTS" = true ] && [ -n "$ARTIFACT_CACHE_URL" ]; then
  # "--deps build" is the whole build plan of squey.bst, minus the element itself
  DEPENDENCIES=$(bst $BUILD_OPTIONS show --deps build --format '%{name}' squey.bst)
  bst $BUILD_OPTIONS artifact push --artifact-remote "$ARTIFACT_CACHE_URL" $DEPENDENCIES || true
fi

# Extract testsuite and code coverage reports out of the build sandbox
if [ "$GITLAB_CI" = true ]; then
  if [ "$CODE_COVERAGE_ENABLED" = true ]; then
    bst $BUILD_OPTIONS artifact log squey.bst | cat # show artifact log to extract code coverage percentage
  fi
  bst $BUILD_OPTIONS artifact checkout squey.bst --no-integrate --ignore-project-artifact-remotes --deps none --hardlinks --directory "${TMP_ARTIFACT_DIR}/build" && cp -r "${TMP_ARTIFACT_DIR}"/build/{junit.xml,code_coverage_report} .. || true
fi
