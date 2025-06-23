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
export SQUEY_PYTHONHOME="$appdir/../Frameworks/Python.framework/Versions/Current"
export SQUEY_PYTHONPATH="$appdir/../Resources/python/site-packages"

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

# Sign libraries an binaries
if [ "$TARGET_TRIPLE" = "x86_64-apple-darwin" ]; then
    find "$testsuitedir" -name "SQUEY_TEST*" -exec install_name_tool -rpath "/mac/lib" "$appdir/../Frameworks" "{}" \; 2> /dev/null
fi
#codesign --force --deep --sign - "$CI_PROJECT_DIR/Squey.app"
find "$appdir/../Frameworks" -name "*.dylib" -exec codesign --force --deep --sign - "{}" \; 2> /dev/null
find "$appdir" -exec codesign --force --deep --sign - "{}" \; 2> /dev/null
#
find "$testsuitedir" -name "SQUEY_TEST*" -exec codesign --force --deep --sign - "{}" \; 2> /dev/null

# Setup Squey config file
configdir="$HOME/.squey"
inifile="$configdir/squey/config.ini"
mkdir -p "$configdir/squey" "${TMPDIR}${USER}"
cp "$CI_PROJECT_DIR/src/pvconfig.ini" "$inifile"
sed -i '' "s|\(nraw_tmp=\).*|\1${TMPDIR}|" "$inifile"

# Increase file descriptors limit to avoid "Too many open files" error
ulimit -n 1048576 

# Run testsuite
ctest_cmd=(ctest --test-dir "$testsuitedir" -j $(nproc) --output-junit "$CI_PROJECT_DIR/junit.xml" --output-on-failure -T test -R 'SQUEY_TEST')
if [ "$TARGET_TRIPLE" = "aarch64-apple-darwin" ]; then
    "${ctest_cmd[@]}"
elif [ "$TARGET_TRIPLE" = "x86_64-apple-darwin" ]; then
    softwareupdate --install-rosetta --agree-to-license || true
    arch -x86_64 bash -c "/usr/sbin/sysctl -a" | grep machdep.cpu.features
    arch -x86_64 bash -c '"$@"' _ "${ctest_cmd[@]}"
fi