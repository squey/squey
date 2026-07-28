#!/bin/sh
set -e
set -x

# Print some info about the environment
sw_vers

# Setup environment variables
packagedir="$CI_PROJECT_DIR/export/$TARGET_TRIPLE"
bundledir="/Volumes/Squey/Squey.app"
appdir="$CI_PROJECT_DIR/Squey.app/Contents/MacOS"
testsuitedir="$CI_PROJECT_DIR/builds/$TARGET_TRIPLE/Clang/RelWithDebInfo"
export PATH="$appdir:/opt/homebrew/bin:$PATH"
export DYLD_LIBRARY_PATH="$appdir/../Frameworks"
export PVKERNEL_PLUGIN_PATH="$appdir/../Frameworks/squey/plugins"
export SQUEY_PLUGIN_PATH="$PVKERNEL_PLUGIN_PATH"
# Qt platform plugins ship inside the bundle while the test binaries are extracted outside
# of it, so Qt does not find them next to the running executable.
export QT_QPA_PLATFORM_PLUGIN_PATH="$appdir/../PlugIns/platforms"
export SQUEY_PYTHONHOME="$appdir/../Frameworks/Python.framework/Versions/Current"
export SQUEY_PYTHONPATH="$appdir/../Resources/python/site-packages"
# Same reason as the Qt plugins above: tshark_path() falls back to looking next to
# the running executable, which holds for the application but not for a test binary
# extracted outside the bundle. Without this the pcap import spawns a tshark that is
# not there, produces no csv, and import_pcap sits until the 300 s test timeout.
export SQUEY_TSHARK_PATH="$appdir/tshark"

# Install dependencies
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ENV_HINTS=1
brew install --formula cmake

# Mount DMG package and extract testsuite
hdiutil attach -nobrowse $packagedir/*.dmg
cp -R -p "$bundledir" "$CI_PROJECT_DIR"
mkdir -p "$testsuitedir"
unzip -qq "$packagedir/testsuite.zip" -d "$testsuitedir"

# Sign libraries and binaries
find "$testsuitedir" -name "SQUEY_TEST*" -exec install_name_tool -rpath "/mac/lib" "$appdir/../Frameworks" "{}" \; 2> /dev/null
find "$appdir/../Frameworks" -name "*.dylib" -exec codesign --force --deep --sign - "{}" \; 2> /dev/null
find "$appdir" -exec codesign --force --deep --sign - "{}" \; 2> /dev/null
find "$testsuitedir" -name "SQUEY_TEST*" -exec codesign --force --deep --sign - "{}" \; 2> /dev/null

# Setup Squey config file
configdir="$HOME/.squey"
inifile="$configdir/squey/config.ini"
mkdir -p "$configdir/squey" "${TMPDIR}${USER}"
cp "$CI_PROJECT_DIR/src/pvconfig.ini" "$inifile"
sed -i '' "s|\(nraw_tmp=\).*|\1${TMPDIR}|" "$inifile"

# Increase file descriptors limit to avoid "Too many open files" error
ulimit -n 1048576

# Run testsuite. ctest only reports "SEGFAULT" for a crashed test, and these
# runners are the only arm64 machines available, so print the crash reports macOS
# writes: they carry the symbolicated stack the log otherwise never shows.
set +e
ctest_status=0
ctest_cmd=(ctest --test-dir "$testsuitedir" -j $(nproc) --output-junit "$CI_PROJECT_DIR/junit.xml" --output-on-failure -T test -R 'SQUEY_TEST')
if [ "$TARGET_TRIPLE" = "aarch64-apple-darwin" ]; then
    "${ctest_cmd[@]}"
    ctest_status=$?
elif [ "$TARGET_TRIPLE" = "x86_64-apple-darwin" ]; then
    softwareupdate --install-rosetta --agree-to-license || true
    arch -x86_64 bash -c "/usr/sbin/sysctl -a" | grep machdep.cpu.features
    arch -x86_64 bash -c '"$@"' _ "${ctest_cmd[@]}"
    ctest_status=$?
fi
set -e

if [ "$ctest_status" -ne 0 ]; then
    # ReportCrash writes the report asynchronously, well after the crashed process
    # is reaped: globbing right here found nothing every time, which is why a
    # SEGFAULT so far reached the log with no stack at all. Give it a chance.
    i=0
    while [ "$i" -lt 30 ]; do
        if ls "$HOME/Library/Logs/DiagnosticReports/"SQUEY_TEST* >/dev/null 2>&1; then
            break
        fi
        sleep 1
        i=$((i + 1))
    done
    for report in "$HOME/Library/Logs/DiagnosticReports/"SQUEY_TEST*; do
        [ -e "$report" ] || continue
        echo "===== crash report: $report ====="
        cat "$report"
    done
fi

exit "$ctest_status"
