#!/bin/bash

# A missing report has to fail the build, otherwise it only surfaces much later
# as an unrelated error in the job publishing it
set -e

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Run from the build directory, where the .gcda files sit next to their objects.
# Anything outside of the source tree is filtered out by --root.
# The filters have to start with a slash: gcovr matches them against absolute
# paths, and prepends the current directory to the ones that do not.
# The coverage threshold is a safety net rather than a quality gate: gcovr exits
# with 0 when it finds no .gcda at all, which would yield an empty report.
mkdir -p code_coverage_report
gcovr \
    --root "$SOURCE_DIR" \
    -j "$(nproc)" \
    --exclude '/.*/build/.*' \
    --exclude '/.*/external/.*' \
    --exclude '/.*/squey-utils/.*' \
    --exclude '/.*/third_party/.*' \
    --exclude '/.*/moc_.*' \
    --fail-under-line 1 \
    --html-nested code_coverage_report/index.html \
    --cobertura code_coverage_report/cobertura-coverage.xml \
    --txt-summary \
    .
