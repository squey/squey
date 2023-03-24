#!/bin/bash

set -e
set -x

function cleanup {
  rm -rf $HOME/.cache/buildstream/artifacts/extract/inendi-inspector/inendi-inspector
  rm -rf $HOME/.cache/buildstream/build
  rm -rf /srv/tmp-inspector/tomjon/*
}

trap cleanup EXIT SIGKILL SIGQUIT SIGSEGV SIGABRT

usage() {
echo "Usage: $0 [--branch=<branch_name_or_tag_name>] [--disable-testsuite] [--cxx_compiler=<g++/clang++>] [--user-target=<USER_TARGET>]"
echo "                  [--workspace-prefix=<prefix>] [--repo=<repository_path>] [--gpg-private-key-path=<key>]"
echo "                  [--gpg-sign-key=<key>] [--code-coverage=<true/false>]" 1>&2; exit 1;

}

# Set default options
BRANCH_NAME=main
BRANCH_SPECIFIED=false
TAG_NAME=
BUILD_TYPE=RelWithDebInfo
CXX_COMPILER=clang++
USER_TARGET=developer
USER_TARGET_SPECIFIED=false
WORKSPACE_PREFIX=
EXPORT_BUILD=false
REPO_DIR="repo"
TESTSUITE_DISABLED=false
GPG_PRIVATE_KEY_PATH=
GPG_SIGN_KEY=
CODE_COVERAGE_ENABLED=false
UPLOAD_DEBUG_SYMBOLS=false

# Override default options with user provided options
OPTS=`getopt -o h:r:m:b:t:c:d:g:k:w:e:p,l,u --long help,flatpak-export:,flatpak-repo:,workspace-prefix:,crash-reporter-token:,gpg-private-key-path:,gpg-sign-key:,branch:,build-type:,cxx_compiler:,user-target:,disable-testsuite:,code-coverage:,upload-debug-symbols: -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ; exit 0 ;;
    -b | --branch ) BRANCH_SPECIFIED=true; BRANCH_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -p | --cxx_compiler ) CXX_COMPILER="$2"; shift 2 ;;
    -m | --user-target ) USER_TARGET_SPECIFIED=true; USER_TARGET="$2"; shift 2 ;;
    -d | --disable-testsuite ) TESTSUITE_DISABLED="$2"; shift 2 ;;
    -w | --workspace-prefix ) WORKSPACE_PREFIX="$2"; shift 2 ;;
    -e | --flatpak-export ) EXPORT_BUILD="$2"; shift 2 ;;
    -r | --flatpak-repo ) REPO_DIR="$2"; shift 2 ;;
    -c | --crash-reporter-token) INSPECTOR_CRASH_REPORTER_TOKEN="$2"; shift 2 ;;
    -g | --gpg-private-key-path ) GPG_PRIVATE_KEY_PATH="$2"; shift 2 ;;
    -k | --gpg-sign-key ) GPG_SIGN_KEY="$2"; shift 2 ;;
    -l | --code-coverage ) CODE_COVERAGE_ENABLED="$2"; shift 2 ;;
    -u | --upload-debug-symbols ) UPLOAD_DEBUG_SYMBOLS="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

source .common.sh

WORKSPACE_NAME="workspace_build"
open_workspace "$WORKSPACE_NAME"

# Use proper branch if specified
ORIGIN="origin"
if git rev-parse "tags/$BRANCH_NAME"; then
    ORIGIN="tags"
fi
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
if [ "$CURRENT_BRANCH" == "HEAD" ] || [ "$BRANCH_SPECIFIED" = "true" ]; then
    pushd .
    cd "$WORKSPACE_PREFIX/$WORKSPACE_NAME"
    git reset --hard HEAD # Clean env
    if [ -n "$CI_PROJECT_PATH" ]; then
      if [ "$CI_MERGE_REQUEST_SOURCE_PROJECT_PATH" != "$CI_PROJECT_PATH" ] && [ -n "$CI_MERGE_REQUEST_SOURCE_PROJECT_PATH" ]; then
          git fetch -a --tags --force
          git checkout $CI_MERGE_REQUEST_TARGET_BRANCH_SHA
          git submodule update --recursive
          git remote set-url origin $CI_MERGE_REQUEST_SOURCE_PROJECT_URL
      else
          git remote set-url origin $CI_PROJECT_URL
      fi
    fi
    git fetch -a --tags --force
    git checkout -B $BRANCH_NAME $ORIGIN/$BRANCH_NAME
    if [ "$CI_MERGE_REQUEST_SOURCE_PROJECT_PATH" == "$CI_PROJECT_PATH" ] || [ -z "$CI_MERGE_REQUEST_SOURCE_PROJECT_PATH" ]; then
      git submodule update --recursive
    fi
    popd
else
    BRANCH_NAME=$CURRENT_BRANCH
fi
if [ $USER_TARGET == "customer" ]; then
    BRANCH_NAME="main"
fi

# Fill-in crash reporter token
if [ ! -z "$INSPECTOR_CRASH_REPORTER_TOKEN" ]; then
  INSPECTOR_CRASH_REPORTER_TOKEN_FILE="$WORKSPACE_PREFIX/$WORKSPACE_NAME/libpvkernel/include/pvkernel/core/PVCrashReporterToken.h"
  sed -e "s|\(INSPECTOR_CRASH_REPORTER_TOKEN\) \"\"|\1 \"$INSPECTOR_CRASH_REPORTER_TOKEN\"|" -i "$INSPECTOR_CRASH_REPORTER_TOKEN_FILE"
fi

# Fill-in release and date
jinja2 -D version="$(cat ../VERSION.txt | tr -d '\n')" -D date="$(date --iso)" files/com.gitlab.inendi.Inspector.metainfo.xml.j2 > files/com.gitlab.inendi.Inspector.metainfo.xml

# Build INENDI Inspector
BUILD_OPTIONS="--option cxx_compiler $CXX_COMPILER"
if [ $USER_TARGET_SPECIFIED = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option user_target $USER_TARGET"
fi
if  [ "$RUN_TESTSUITE" = false ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option disable_testsuite True"
fi
if  [ "$CODE_COVERAGE_ENABLED" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option code_coverage True"
fi
bst $BUILD_OPTIONS build inendi-inspector.bst

# Run testsuite with "bst shell" to have network access (bst hasn't a "test-commands" (yet?) like in flatpak-builder)
if  [ "$TESTSUITE_DISABLED" = "false" ]; then
    bst $BUILD_OPTIONS shell $MOUNT_OPTS inendi-inspector.bst -- bash -c " \
    cp --preserve -r /compilation/* .
    TESTS=\"-R INSPECTOR_TEST\"
    if [ $CODE_COVERAGE_ENABLED = true ]; then CODE_COVERAGE_COMMAND=\"-T coverage\"; TESTS=\"-R 'INSPECTOR_TEST|PVCOP_TEST'\"; fi
    cd build && run_cmd.sh ctest --output-on-failure -T test \${CODE_COVERAGE_COMMAND} \${TESTS} || if [ $CODE_COVERAGE_ENABLED = false ]; then exit 1; fi
    # Generate code coverage report
    if [ $CODE_COVERAGE_ENABLED = true ]; then
        ./scripts/gen_code_coverage_report.sh
        cp -r code_coverage_report /srv/tmp-inspector
    fi" || exit 1 # fail the testsuite on errors
fi

# Upload debug symbols
if  [ "$UPLOAD_DEBUG_SYMBOLS" = true ]; then
  VERSION="$(cat $WORKSPACE_PREFIX/$WORKSPACE_NAME/VERSION.txt)"
  bst $BUILD_OPTIONS shell $MOUNT_OPTS inendi-inspector.bst -- bash -c " \
      SYM_DIR=\"/tmp/inendi-inspector.sym.d\"
      rm -rf \"\$SYM_DIR\" && mkdir -p \"\$SYM_DIR\"
      cd /compilation/build
      find . -type f \( -name *.so* -o -name \"inendi-inspector\" \) -exec sh -c 'dump_syms \"\$0\" > \"\$1\"/\"\$(basename \"\$0\").sym\"' \"{}\" \"\$SYM_DIR\" \;
      find \"\$SYM_DIR\" -type f -exec sed 's|/buildstream/inendi-inspector/inendi-inspector.bst/||' -i \"{}\" \;
      find \"\$SYM_DIR\" -type f -exec sym_upload \"{}\" \"https://inendi_inspector.bugsplat.com/post/bp/symbol/breakpadsymbols.php?appName=INENDI%20Inspector&appVer=$VERSION\" \;
      "
fi

# Export flatpak images
if [ "$EXPORT_BUILD" = true ]; then
  if [[ ! -z "$GPG_PRIVATE_KEY_PATH" ]]; then
    # Import GPG private key
    gpg --import --no-tty --batch --yes $GPG_PRIVATE_KEY_PATH
  fi

  # Export flatpak Release image
  rm -rf $DIR/build
  bst $BUILD_OPTIONS build flatpak/com.gitlab.inendi.Inspector.bst
  bst $BUILD_OPTIONS checkout flatpak/com.gitlab.inendi.Inspector.bst "$DIR/build"
  if [[ ! -z "$GPG_SIGN_KEY" ]]; then
    flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  else
    flatpak build-export --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  fi

  ## Export flatpak Debug image
  #rm -rf $DIR/build
  #bst $BUILD_OPTIONS build flatpak/com.gitlab.inendi.Inspector.Debug.bst
  #bst $BUILD_OPTIONS checkout flatpak/com.gitlab.inendi.Inspector.Debug.bst "$DIR/build"
  #if [[ ! -z "$GPG_SIGN_KEY" ]]; then
  #  flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  #else
  #  flatpak build-export --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  #fi
fi

# Push artifacts
bst --option push_artifacts True push `ls elements -p -I "base.bst" -I "freedesktop-sdk.bst" -I "inendi-inspector*.bst" |grep -v / | tr '\n' ' '` || true
