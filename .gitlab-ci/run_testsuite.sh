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
    # These runners write no DiagnosticReports at all -- waiting 30 s for one to
    # appear yielded nothing -- and this Qt prints no backtrace of its own, so a
    # SEGFAULT reaches the log as that single word. Replay each failed test under
    # lldb instead: it is the only way to get a symbolicated arm64 stack, and the
    # tests are short enough to afford a second run.
    # -n: never prompt, a password prompt here would hang the job.
    sudo -n DevToolsSecurity -enable > /dev/null 2>&1 || true
    # "-T test" writes a timestamped variant, so take the newest of both forms.
    failed_log=$(ls -t "$testsuitedir/Testing/Temporary/"LastTestsFailed*.log 2>/dev/null | head -1)
    if [ -n "$failed_log" ] && [ -r "$failed_log" ]; then
        # Lines read "<index>:<test name>"; keep the crashed ones affordable.
        for test_name in $(sed 's/^[0-9]*://' "$failed_log" | head -3); do
            echo "===== lldb backtrace: $test_name ====="
            # The command and working directory come from ctest itself, so the
            # test is replayed exactly as it ran, environment included.
            ctest --test-dir "$testsuitedir" -R "^${test_name}\$" --show-only=json-v1 \
                > /tmp/test_def.json 2>/dev/null || continue
            python3 - "$test_name" <<'PY' > /tmp/replay.sh || continue
import json, shlex, sys
with open("/tmp/test_def.json") as f:
    tests = json.load(f).get("tests", [])
if not tests:
    raise SystemExit(1)
t = tests[0]
cwd, env = None, []
for p in t.get("properties", []):
    if p["name"] == "WORKING_DIRECTORY":
        cwd = p["value"]
    elif p["name"] == "ENVIRONMENT":
        env = p["value"] if isinstance(p["value"], list) else [p["value"]]
if cwd:
    print("cd %s" % shlex.quote(cwd))
for e in env:
    print("export %s" % shlex.quote(e))
print("lldb -b -o run -o 'bt all' -o quit -- %s"
      % " ".join(shlex.quote(a) for a in t["command"]))
PY
            sh /tmp/replay.sh 2>&1 | tail -80 || true
        done
    fi
fi

exit "$ctest_status"
