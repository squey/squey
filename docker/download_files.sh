#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${DIR}/.env.conf"

install_mode="$1"
if [ "$#" -eq 0 ]
then
    mkdir "${DATA_PATH}" &> /dev/null

    command -v flatpak &> /dev/null || { echo >&2 "'flatpak' executable is required to execute this script."; exit 1; }
    command -v wget &> /dev/null || { echo >&2 "'wget' executable is required to execute this script."; exit 1; }

    if [[ `flatpak remote-list |grep flathub |grep -v -q system` == 1 ]]; then
        flatpak remote-add --user --if-not-exists flathub "${FLATHUB_REPO_FLATPAKREF}"
    fi

    FLATPAK_SYSTEM_REPO_DIR="/var/lib/flatpak/repo"
    FLATPAK_USER_REPO_DIR="~/.local/share/flatpak/repo"

    # Export Freedesktop runtime bundle
    echo "[1/2] Exporting Flatpak runtime bundle ..."
    flatpak_install_type=$(flatpak info "$RUNTIME_NAME//$RUNTIME_BRANCH" | grep "Installation: " | cut -d " " -f 2)
    runtime_not_installed=$?
    if [ $flatpak_install_type == "user" ]; then
        USER_OPT="--user"
        FLATPAK_REPO_DIR="$FLATPAK_USER_REPO_DIR"
    else
        FLATPAK_REPO_DIR="$FLATPAK_SYSTEM_REPO_DIR"
    fi
    if [ $runtime_not_installed -eq 1 ]
    then
        flatpak install $USER_OPT -y flathub "$RUNTIME_NAME//$RUNTIME_BRANCH"  &> /dev/null
    else
        flatpak update $USER_OPT -y "$RUNTIME_NAME//$RUNTIME_BRANCH" &> /dev/null
    fi
    flatpak build-bundle --runtime $FLATPAK_REPO_DIR "${DATA_PATH}/runtime.flatpak" "$RUNTIME_NAME" "$RUNTIME_BRANCH"
    if [ $runtime_not_installed -eq 1 ]
    then
        flatpak uninstall $USER_OPT -y "$RUNTIME_NAME//$RUNTIME_BRANCH" &> /dev/null
    fi

    # Export Squey bundle
    echo "[2/2] Exporting Squey bundle ..."
    flatpak info "${SQUEY_NAME}"  &> /dev/null
    squey_not_installed=$?
    if [ $squey_not_installed  -eq 1 ]
    then
        flatpak install $USER_OPT -y flathub "${SQUEY_NAME}" &> /dev/null
    else
        flatpak update $USER_OPT -y "${SQUEY_NAME}" &> /dev/null
    fi
    flatpak build-bundle --repo-url="${FLATHUB_REPO}" --runtime-repo="${FLATHUB_REPO_FLATPAKREF}" $FLATPAK_REPO_DIR "${DATA_PATH}/squey.flatpak" "${SQUEY_NAME}" "stable"
    if [ $squey_not_installed -eq 1 ]
    then
        flatpak uninstall $USER_OPT -y "${SQUEY_NAME}" &> /dev/null
    fi

    # Download NICE DCV
    wget $NICE_DCV_URL -P "${DATA_PATH}"
fi

if [ "${install_mode}" == "online" ]
then
    wget $NICE_DCV_URL -P "${INSTALL_PATH}"
fi
