#!/bin/bash
#
# Gives the container a way to the desktop portal of the host: a proxy of the
# session bus that lets nothing else through, whose socket
# .devcontainer/devcontainer.json mounts.
#
# Links clicked in the container open in the browser of the host this way:
# xdg-open in the image is the one of flatpak-xdg-utils, which asks the portal
# over the session bus. Squey reads the light or dark scheme of the desktop from
# the portal too. Without a session bus, both give up.
#
# Not the whole bus, which would let the container drive the session: start any
# command on the host through systemd --user, for one. Nor the whole portal, as
# flatpak gives its applications: the portal takes whatever comes through this
# proxy for a program of the host, which it trusts more than a sandboxed one.
# Only the methods named below get through.
#
# Runs on the host, from prepare_gpu.sh. One proxy serves every container, and
# outlives them. It is started again when none runs with the arguments below,
# as after logging in again, or once they change.
#
# Finding no session bus or no xdg-dbus-proxy is not an error: the container
# starts either way, and links simply do not open from it.

set -e

# The path .devcontainer/devcontainer.json names, which cannot follow
# XDG_CACHE_HOME. A directory rather than the socket itself, which a proxy
# started again replaces: a running container then reaches the new one.
PORTAL="$HOME/.cache/squey-devcontainer/portal"

# Mounted whether a proxy listens in it or not: a missing source would keep the
# container from starting.
mkdir -p "$PORTAL"

if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
    echo "No session bus; links will not open from the container." >&2
    exit 0
fi
if ! command -v xdg-dbus-proxy > /dev/null; then
    echo "No xdg-dbus-proxy; links will not open from the container." >&2
    echo "Install it with the package manager of this machine; flatpak depends on it." >&2
    exit 0
fi

DESKTOP=/org/freedesktop/portal/desktop
PROXY=(xdg-dbus-proxy "$DBUS_SESSION_BUS_ADDRESS" "$PORTAL/bus" --filter
    # What xdg-open calls
    "--call=org.freedesktop.portal.Desktop=org.freedesktop.portal.OpenURI.*@$DESKTOP"
    # What PVCore::PVTheme calls, the introspection QDBusInterface starts with,
    # and the signal it waits for
    "--call=org.freedesktop.portal.Desktop=org.freedesktop.portal.Settings.*@$DESKTOP"
    "--call=org.freedesktop.portal.Desktop=org.freedesktop.DBus.Introspectable.Introspect@$DESKTOP"
    "--broadcast=org.freedesktop.portal.Desktop=org.freedesktop.portal.Settings.SettingChanged@$DESKTOP"
)

wanted="$(printf '%s\n' "${PROXY[@]}")"
running=false
for pid in $(pgrep -u "$UID" -x xdg-dbus-proxy); do
    arguments="$(tr '\0' '\n' < "/proc/$pid/cmdline" 2> /dev/null)" || continue
    if [ "$arguments" = "$wanted" ] && [ -S "$PORTAL/bus" ]; then
        running=true
    elif grep -qxF -- "$PORTAL/bus" <<< "$arguments"; then
        # Started by another version of this script or for another session, or
        # its socket is gone. Waited for, so that it cannot remove the socket
        # of the new one on its way out.
        kill "$pid" 2> /dev/null || true
        for _ in {1..50}; do
            kill -0 "$pid" 2> /dev/null || break
            sleep 0.1
        done
    fi
done

if ! $running; then
    # Left by a proxy that ended, it would keep the new one from listening.
    rm -f "$PORTAL/bus"
    # In a session of its own, so that whatever runs this script cannot take it
    # down on the way out
    setsid "${PROXY[@]}" < /dev/null > /dev/null 2>&1 &
    echo "Desktop portal proxied for the container."
fi
