#!/bin/bash

tmp_output_file="/tmp/$(echo `basename "$0"`.$$)"

function cleanup {
  rm -rf "$tmp_output_file" &> /dev/null
}

trap cleanup EXIT SIGKILL SIGQUIT SIGSEGV SIGABRT

nvidia_drivers=$(flatpak --gl-drivers | grep nvidia)

if [ ! -z $nvidia_drivers ]; then
    flatpak_nvidia_drivers="org.freedesktop.Platform.GL."$nvidia_drivers
    if [[ $(flatpak info "$flatpak_nvidia_drivers" &>/dev/null) == 0 ]]; then
        exit 0
    fi

    # Install NVIDIA drivers flatpak package
    flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
    bash -i -c "flatpak install --user -y flathub $flatpak_nvidia_drivers" &> "$tmp_output_file" &

    # Handle progress bar dialog
    dbus_process=$(kdialog --title "NVIDIA drivers" --progressbar "Installing $flatpak_nvidia_drivers ...")
    while ! $(flatpak info "$flatpak_nvidia_drivers" &> /dev/null);
    do
        percentage=$(tail -n 1 "$tmp_output_file" | strings | tail -n 2 | grep --line-buffered -o '[^ ]*%')
        qdbus $dbus_process Set org.kde.kdialog.ProgressDialog value ${percentage%?} &> /dev/null
        sleep 1
    done
    qdbus $dbus_process close > /dev/null;
fi