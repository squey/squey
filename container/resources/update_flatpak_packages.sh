#!/bin/bash

tmp_output_file="/tmp/$(basename "$0").output"
tmp_pid_file="/tmp/$(basename "$0").pid.$$"
tmp_cmdline_file="/tmp/$(basename "$0").cmdline"

function process_running {
    pid="$1"
    cmdline="$2"
    [ -f "$tmp_output_file" ] && [ -n "$pid" ] && [ -d "/proc/$pid" ] && [ "$(tr -d '\0' < /proc/$pid/cmdline)" == "$cmdline" ]
}

function ongoing_installation {
    pid_path=$(ls "${tmp_pid_file%.*}."* 2> /dev/null)
    if [ -n "$pid_path" ] ; then
        owner_pid="${pid_path##*.}"
        if [ "$owner_pid" = "$$" ]; then
            return 1 # this script is doing the installation
        else
            pid=$(cat "$pid_path")
            cmdline=$(cat "$tmp_cmdline_file")
            if process_running "$pid" "$cmdline"; then
                return 0 # another script is doing the installation
            else
                return 1 # some garbage file was left behind
            fi
        fi
    fi
    return 1 # no ongoing installation
}

function cleanup {
    rm -rf "$tmp_output_file" &> /dev/null
    rm -rf "$tmp_pid_file" &> /dev/null
    rm -rf "$tmp_cmdline_file" &> /dev/null
}

function get_package_count {
    package_count=$(grep -P '\d+\.\t' "$tmp_output_file" | cut -f 3,4 | wc -l)
    echo "$package_count"
}

function get_package_name_from_index {
    package_index="$1"
    package_name=$(grep -P '\d+\.\t' "$tmp_output_file" | cut -f 3,4 | sed 's|\t|//|' | sed -n "$package_index"p)
    echo "$package_name"
}

function get_current_package_index {
    local index
    while [ -z "$index" ] && [ -f "$tmp_output_file" ]; do
        res=$(tail -n 1 "$tmp_output_file" | strings | tail -n 2 | grep -E "Installing|Updating" | sed -n 1p | cut -d " " -f 2)
        if grep -q "?" <<< "$res" ; then
            index="1"
        else
            index="$(echo "$res"| cut -d "/" -f 1)"
        fi
    done
    echo "$index"
}

function monitor_package_installation {
    # Handle progress bar dialog
    package_index="$1"
    pid="$2"
    cmdline="$3"
    package_name=$(get_package_name_from_index "$package_index")
    package_count=$(get_package_count)
    dbus_process=$(kdialog --title "Updating Squey" --progressbar "Updating flatpak package [$package_index/$package_count] : $package_name")
    while process_running "$pid" "$cmdline" && [ "$package_index" = "$(get_current_package_index)" ] ; do
        percentage=$(tail -n 1 "$tmp_output_file" | strings | tail -n 2 | grep --line-buffered -o '[^ ]*%')
        qdbus $dbus_process Set org.kde.kdialog.ProgressDialog value "${percentage%?}" &> /dev/null
        sleep 1
    done
    qdbus $dbus_process close > /dev/null;
}

if ! ongoing_installation ; then
    trap cleanup EXIT SIGQUIT SIGSEGV SIGABRT

    # Check if NVIDIA flatpak drivers needs to be installed/updated
    nvidia_drivers=$(flatpak --gl-drivers | grep -i nvidia)
    if [ -n $nvidia_drivers ]; then
        flatpak_nvidia_drivers="org.freedesktop.Platform.GL.$nvidia_drivers"
        if ! flatpak info "$flatpak_nvidia_drivers" &>/dev/null; then
            flatpak_commands+=( "sudo flatpak install -y flathub $flatpak_nvidia_drivers" )
        fi
    fi

    # Check if Squey flatpak package needs to be updated
    squey_flatpak_package="org.squey.Squey"
    flatpak remote-ls --updates | grep -q "$squey_flatpak_package"
    if [ "$?" = "0" ]; then
        flatpak_commands+=( "sudo flatpak update --no-related -y $squey_flatpak_package" )
    fi
    
    # Run flatpak commands
    for flatpak_command in "${flatpak_commands[@]}"; do
        bash -c "$flatpak_command" > "$tmp_output_file" &
        pid=$!
        sleep 1
        cmdline="$(tr -d '\0' < /proc/$pid/cmdline)"
        echo "$pid" > "$tmp_pid_file"
        echo "$cmdline" > "$tmp_cmdline_file"

        while process_running "$pid" "$cmdline"; do
            current_package_index=$(get_current_package_index)
            monitor_package_installation "$current_package_index" "$pid" "$cmdline"
        done
    done

    sudo flatpak uninstall --unused -y
else
    while [ -f "$tmp_output_file" ] ; do
        owner_pid_path=$(ls "${tmp_pid_file%.*}."* 2> /dev/null)
        pid=$(cat "$owner_pid_path")
        cmdline=$(cat "$tmp_cmdline_file")
        current_package_index=$(get_current_package_index)
        monitor_package_installation "$current_package_index" "$pid" "$cmdline"
    done
fi


