#!/bin/sh

# Check flatpak update
latest_commit=`su - squey -s "/bin/ash" -c "flatpak --user remote-info flathub $1 | sed -n -r 's/.*Commit: (.*)/\1/p'"`
current_commit=`su - squey -s "/bin/ash" -c "flatpak --user info $1 | sed -n -r 's/.*Commit: (.*)/\1/p'"`

if [[ "$current_commit" != "$latest_commit" ]]; then
    # Update Squey flatpak package
    su - squey -s "/bin/ash" -c "flatpak --user update -y --noninteractive"
    su - squey -s "/bin/ash" -c "flatpak --user uninstall -y --unused"

    # Update Alpine Linux
    apk upgrade --update-cache --available
fi
