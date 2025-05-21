#!/bin/bash

TMP_ARTIFACT_DIR="$(mktemp -d)"

function cleanup {
  rm -rf -- "${TMP_ARTIFACT_DIR}"
  rm -rf $HOME/.cache/buildstream/artifacts/extract/squey/squey
  rm -rf $HOME/.cache/buildstream/build
  rm -rf /srv/tmp-squey/tomjon/*
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
if  [ "$CODE_COVERAGE_ENABLED" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option code_coverage True"
  pushd .
  cd ../src/libpvcop/tests/files
  git submodule update --checkout
  popd
fi

if [ "$EXPORT_BUILD" = false ]; then
  bst $BUILD_OPTIONS build squey.bst
elif [ "$TARGET_TRIPLE" == "x86_64-linux-gnu" ]; then # Generate Linux flatpak repository

  if [[ ! -z "$GPG_PRIVATE_KEY_PATH" ]]; then
    # Import GPG private key
    gpg --import --no-tty --batch --yes $GPG_PRIVATE_KEY_PATH
  fi

  # Export flatpak Release image
  bst $BUILD_OPTIONS build flatpak/org.squey.Squey.bst
  if  [ "$UPLOAD_DEBUG_SYMBOLS" = true ] ; then   # Upload debug symbols
    VERSION="$(cat ../VERSION.txt)"
    bst $BUILD_OPTIONS shell $MOUNT_OPTS squey.bst -- bash -c " \
        SYM_DIR=\"/tmp/squey.sym.d\"
        rm -rf \"\$SYM_DIR\" && mkdir -p \"\$SYM_DIR\"
        cd /compilation_build
        find . -type f \( -name *.so* -o -name \"squey\" \) -exec sh -c 'dump_syms \"\$0\" > \"\$1\"/\"\$(basename \"\$0\").sym\"' \"{}\" \"\$SYM_DIR\" \;
        find \"\$SYM_DIR\" -type f -exec sed 's|/buildstream/squey/squey.bst/||' -i \"{}\" \;
        find \"\$SYM_DIR\" -type f -exec sym_upload \"{}\" \"https://squey.bugsplat.com/post/bp/symbol/breakpadsymbols.php?appName=Squey&appVer=$VERSION\" \;
        "
  fi
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
  rm -rf "$EXPORT_DIR/$TARGET_TRIPLE"
  mkdir -p "$EXPORT_DIR/$TARGET_TRIPLE"
  bst $BUILD_OPTIONS shell -b --mount "$EXPORT_DIR/$TARGET_TRIPLE" /output macos_bundle/dmg-image.bst bash "buildstream/files/macos_bundle/make-dmg-image.sh"
elif [ "$TARGET_TRIPLE" == "x86_64-w64-mingw32" ]; then # Generate Windows MSIX package
  bst $BUILD_OPTIONS build msix_package/msix-package.bst
  rm -rf "$EXPORT_DIR/$TARGET_TRIPLE"
  mkdir -p "$EXPORT_DIR/$TARGET_TRIPLE"
  bst $BUILD_OPTIONS shell -b --mount "$EXPORT_DIR/$TARGET_TRIPLE" /output msix_package/msix-package.bst bash "buildstream/files/msix_package/make-msix-package.sh"
fi

# Push artifacts
if [ "$PUSH_ARTIFACTS" = true ] && [ "$CODE_COVERAGE_ENABLED" = false ]; then
  bst $BUILD_OPTIONS --option push_artifacts True artifact push `ls elements -p -I "base.bst" -I "freedesktop-sdk.bst" -I "squey*.bst" |grep -v / | tr '\n' ' '` || true
fi

# Extract testsuite and code coverage reports out of the build sandbox
if [ "$GITLAB_CI" = true ]; then
  if [ "$CODE_COVERAGE_ENABLED" = true ]; then
    bst $BUILD_OPTIONS artifact log squey.bst | cat # show artifact log to extract code coverage percentage
  fi
  bst $BUILD_OPTIONS artifact checkout squey.bst --no-integrate --ignore-project-artifact-remotes --deps none --hardlinks --directory "${TMP_ARTIFACT_DIR}/build" && cp -r "${TMP_ARTIFACT_DIR}"/build/{junit.xml,code_coverage_report} .. || true
fi