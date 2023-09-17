#!/bin/bash

set -e
set -x

function cleanup {
  rm -rf $HOME/.cache/buildstream/artifacts/extract/squey/squey
  rm -rf $HOME/.cache/buildstream/build
  rm -rf /srv/tmp-squey/tomjon/*
}

trap cleanup EXIT SIGKILL SIGQUIT SIGSEGV SIGABRT

usage() {
echo "Usage: $0 [--branch=<branch_name_or_tag_name>] [--disable-testsuite=<true/false>] [--cxx_compiler=<g++/clang++>] [--user-target=<USER_TARGET>]"
echo "                  [--flatpak-export=<true/false>] [--flatpak-repo=<repository_path>] [--gpg-private-key-path=<key>] [--gpg-sign-key=<key>] "
echo "                  [--code-coverage=<true/false>]" 1>&2; exit 1;

}

# Set default options
BRANCH_NAME=main
TAG_NAME=
BUILD_TYPE=RelWithDebInfo
CXX_COMPILER=clang++
USER_TARGET=developer
USER_TARGET_SPECIFIED=false
EXPORT_BUILD=false
REPO_DIR="repo"
TESTSUITE_DISABLED=false
GPG_PRIVATE_KEY_PATH=
GPG_SIGN_KEY=
CODE_COVERAGE_ENABLED=false
UPLOAD_DEBUG_SYMBOLS=false

# Override default options with user provided options
OPTS=`getopt -o h:r:m:b:t:d:g:k:e:p,l,u --long help,flatpak-export:,flatpak-repo:,gpg-private-key-path:,gpg-sign-key:,branch:,build-type:,cxx_compiler:,user-target:,disable-testsuite:,code-coverage:,upload-debug-symbols: -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ; exit 0 ;;
    -b | --branch ) BRANCH_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -p | --cxx_compiler ) CXX_COMPILER="$2"; shift 2 ;;
    -m | --user-target ) USER_TARGET_SPECIFIED=true; USER_TARGET="$2"; shift 2 ;;
    -d | --disable-testsuite ) TESTSUITE_DISABLED="$2"; shift 2 ;;
    -e | --flatpak-export ) EXPORT_BUILD="$2"; shift 2 ;;
    -r | --flatpak-repo ) REPO_DIR="$2"; shift 2 ;;
    -g | --gpg-private-key-path ) GPG_PRIVATE_KEY_PATH="$2"; shift 2 ;;
    -k | --gpg-sign-key ) GPG_SIGN_KEY="$2"; shift 2 ;;
    -l | --code-coverage ) CODE_COVERAGE_ENABLED="$2"; shift 2 ;;
    -u | --upload-debug-symbols ) UPLOAD_DEBUG_SYMBOLS="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

source .common.sh

# Fill-in release and date
CHANGELOG_CONTENT="$(awk '/^---.*---$/ {count++} count == 1 {print} count == 2 {exit}' ../CHANGELOG | head -n -2 | tail -n +3)"
CHANGELOG_CONTENT="$(echo "$CHANGELOG_CONTENT" | sed '1 s|\(.*\)|<p>\1</p>\n          <ul>|' | sed 's|\* \(.*\)|            <li>\1</li>|' | (tee -a && echo '          </ul>'))"
jinja2 -D version="$(cat ../VERSION.txt | tr -d '\n')" -D date="$(date --iso)" -D changelog="$CHANGELOG_CONTENT" files/org.squey.Squey.metainfo.xml.j2 > files/org.squey.Squey.metainfo.xml

# Build Squey
BUILD_OPTIONS="--option cxx_compiler $CXX_COMPILER"
if [ $USER_TARGET_SPECIFIED = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option user_target $USER_TARGET"
fi
if  [ "$TESTSUITE_DISABLED" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option disable_testsuite True"
fi
if  [ "$CODE_COVERAGE_ENABLED" = true ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option code_coverage True"
fi
bst $BUILD_OPTIONS build squey.bst

# Run testsuite with "bst shell" to have network access (bst hasn't a "test-commands" (yet?) like in flatpak-builder)
if  [ "$TESTSUITE_DISABLED" = "false" ]; then
    bst $BUILD_OPTIONS shell $MOUNT_OPTS squey.bst -- bash -c " \
    cp --preserve -r /compilation/* .
    TESTS=\"-R SQUEY_TEST\"
    if [ $CODE_COVERAGE_ENABLED = true ]; then CODE_COVERAGE_COMMAND=\"-T coverage\"; TESTS=\"-R 'SQUEY_TEST|PVCOP_TEST'\"; fi
    cd build && run_cmd.sh ctest --output-junit /srv/tmp-squey/junit.xml --output-on-failure -T test \${CODE_COVERAGE_COMMAND} \${TESTS} || if [ $CODE_COVERAGE_ENABLED = false ]; then exit 1; fi
    # Generate code coverage report
    if [ $CODE_COVERAGE_ENABLED = true ]; then
        ./scripts/gen_code_coverage_report.sh
        cp -r code_coverage_report /srv/tmp-squey
    fi" || exit 1 # fail the testsuite on errors
fi

# Upload debug symbols
if  [ "$UPLOAD_DEBUG_SYMBOLS" = true ]; then
  VERSION="$(cat ../VERSION.txt)"
  bst $BUILD_OPTIONS shell $MOUNT_OPTS squey.bst -- bash -c " \
      SYM_DIR=\"/tmp/squey.sym.d\"
      rm -rf \"\$SYM_DIR\" && mkdir -p \"\$SYM_DIR\"
      cd /compilation/build
      find . -type f \( -name *.so* -o -name \"squey\" \) -exec sh -c 'dump_syms \"\$0\" > \"\$1\"/\"\$(basename \"\$0\").sym\"' \"{}\" \"\$SYM_DIR\" \;
      find \"\$SYM_DIR\" -type f -exec sed 's|/buildstream/squey/squey.bst/||' -i \"{}\" \;
      find \"\$SYM_DIR\" -type f -exec sym_upload \"{}\" \"https://squey.bugsplat.com/post/bp/symbol/breakpadsymbols.php?appName=Squey&appVer=$VERSION\" \;
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
  bst $BUILD_OPTIONS build flatpak/org.squey.Squey.bst
  bst $BUILD_OPTIONS artifact checkout flatpak/org.squey.Squey.bst --directory "$DIR/build"
  if [[ ! -z "$GPG_SIGN_KEY" ]]; then
    flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  else
    flatpak build-export --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  fi

  ## Export flatpak Debug image
  #rm -rf $DIR/build
  #bst $BUILD_OPTIONS build flatpak/org.squey.Squey.Debug.bst
  #bst $BUILD_OPTIONS checkout flatpak/org.squey.Squey.Debug.bst "$DIR/build"
  #if [[ ! -z "$GPG_SIGN_KEY" ]]; then
  #  flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  #else
  #  flatpak build-export --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  #fi
fi

# Push artifacts
bst --option push_artifacts True artifact push `ls elements -p -I "base.bst" -I "freedesktop-sdk.bst" -I "squey*.bst" |grep -v / | tr '\n' ' '` || true
