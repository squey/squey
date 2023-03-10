#!/bin/bash

tmp_output_file="/tmp/$(basename "$0").$$"

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
    while [ -z "$index" ]; do
        index=$(tail -n 1 "$tmp_output_file" | strings | tail -n 2 | grep -E "Installing|Updating" | sed -n 1p | cut -d " " -f 2 | cut -d "/" -f 1)
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
    dbus_process=$(kdialog --title "Updating INENDI Inspector" --progressbar "Updating flatpak package [$package_index/$package_count] : $package_name")
    while process_running "$pid" "$cmdline" && [ "$package_index" = "$(get_current_package_index)" ] ; do
        percentage=$(tail -n 1 "$tmp_output_file" | strings | tail -n 2 | grep --line-buffered -o '[^ ]*%')
        qdbus $dbus_process Set org.kde.kdialog.ProgressDialog value "${percentage%?}" &> /dev/null
        sleep 1
    done
    qdbus $dbus_process close > /dev/null;
}

function process_running {
    pid="$1"
    cmdline="$2"
    [ -n "${pid}" ] && [ -d "/proc/${pid}" ] && [ "$(tr -d '\0' < /proc/$pid/cmdline)" == "$cmdline" ]
}

function ongoing_installation {
    pid_path=$(ls "${tmp_output_file%.*}."* 2> /dev/null)
    if [ -n "$pid_path" ] ; then
        pid="${pid_path##*.}"
        if [ -z "${pid}" ] || [ ! -d "/proc/${pid}" ]; then
            rm -rf "$pid_path" # cleanup garbage file
            return 1
        else
            return 0
        fi
    fi
    return 1
}

function cleanup {
    if ! ongoing_installation ; then
        rm -rf "$tmp_output_file" &> /dev/null
    fi
}

trap cleanup EXIT SIGQUIT SIGSEGV SIGABRT

# Check if there is any update for INENDI Inspector
flatpak remote-ls --updates | grep -q com.gitlab.inendi.Inspector
update_inspector=$([ $? = 0 ] && echo "true" || echo "false")

if [ "$update_inspector" = "true" ]; then
    if ! ongoing_installation ; then
        sudo flatpak update -y | tee "$tmp_output_file" & # Update all flatpak packages
        pid=$!
    else
        pid_path=$(ls "${tmp_output_file%.*}."* 2> /dev/null)
        pid="${pid_path##*.}"
        tmp_output_file="${tmp_output_file%.*}.$pid"
    fi
    cmdline=$(tr -d '\0' < /proc/$pid/cmdline)

    while process_running "$pid" "$cmdline"; do
        current_package_index=$(get_current_package_index)
        monitor_package_installation "$current_package_index" "$pid" "$cmdline"
    done

    if ! ongoing_installation ; then
        sudo flatpak uninstall --unused -y
    fi
fi
