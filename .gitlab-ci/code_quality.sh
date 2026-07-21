#!/bin/bash
set -e

# Run a static analysis of the C++ code base and export the findings both as a
# GitLab Code Quality report and as a browsable HTML report published on Pages.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Some of these directories live in submodules which may not have been fetched
SOURCES=""
for dir in src/include src/furl src/gui-qt/src src/libpvdisplays src/libpvguiqt/src \
           src/libpvkernel/src src/libpvparallelview/src src/libsquey/src src/squey-utils; do
    [ -d "$dir" ] && SOURCES="$SOURCES $dir"
done

# Headers of the whole project, so that cppcheck does not give up on types it
# cannot resolve (libpvcop alone is worth ~50% more findings on libsquey)
INCLUDES=""
for dir in src/include src/*/include; do
    INCLUDES="$INCLUDES -I $dir"
done

# Third party code and test suites are not ours to fix
EXCLUDES=""
for dir in $(find $SOURCES -type d \( -name tests -o -name third_party -o -name external \)); do
    EXCLUDES="$EXCLUDES -i $dir"
done

cppcheck \
    --enable=all \
    --inline-suppr \
    --library=qt \
    --relative-paths="$PWD" \
    -j "$(nproc)" \
    $INCLUDES \
    $EXCLUDES \
    --suppress=missingInclude \
    --suppress=missingIncludeSystem \
    --suppress=unusedFunction \
    --suppress=unmatchedSuppression \
    --suppress=checkersReport \
    --suppress=normalCheckLevelMaxBranches \
    --suppress=noValidConfiguration \
    --suppress=toomanyconfigs \
    --xml --xml-version=2 \
    $SOURCES 2> cppcheck.xml

python3 .gitlab-ci/cppcheck_to_codequality.py cppcheck.xml gl-code-quality-report.json
cppcheck-htmlreport --file=cppcheck.xml --report-dir=code_quality_report --source-dir=.
