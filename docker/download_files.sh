#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${DIR}/.env.conf"

install_mode="$1"
if [ "$#" -eq 0 ]
then

    command -v flatpak &> /dev/null || { echo >&2 "'flatpak' executable is required to execute this script."; exit 1; }
    command -v wget &> /dev/null || { echo >&2 "'wget' executable is required to execute this script."; exit 1; }
    command -v aws &> /dev/null || { echo >&2 "'aws' executable is required to execute this script."; exit 1; }

    flatpak remote-add --user --if-not-exists flathub "${FLATHUB_REPO}"
    flatpak remote-add --user --if-not-exists --no-gpg-verify inendi_tmp https://repo.esi-inendi.com/flatpak/

    # Export Freedesktop runtime bundle
    echo "[1/3] Exporting Flatpak runtime bundle ..."
    flatpak info "$RUNTIME_NAME//$RUNTIME_BRANCH" &> /dev/null
    runtime_not_installed=$?
    if [ $runtime_not_installed -eq 1 ]
    then
        flatpak install --user -y flathub "$RUNTIME_NAME//$RUNTIME_BRANCH"  &> /dev/null
    else
        flatpak update --user -y "$RUNTIME_NAME//$RUNTIME_BRANCH" &> /dev/null
    fi
    flatpak build-bundle --runtime --runtime-repo=${FLATHUB_REPO} ~/.local/share/flatpak/repo "${INSTALL_PATH}/runtime.flatpak" "$RUNTIME_NAME" "$RUNTIME_BRANCH"
    if [ $runtime_not_installed -eq 1 ]
    then
        flatpak uninstall --user -y "$RUNTIME_NAME//$RUNTIME_BRANCH" &> /dev/null
    fi

    # Export NVIDIA drivers bundle
    echo "[2/3] Exporting NVIDIA drivers bundle ..."
    if [ $runtime_not_installed -eq 1 ]
    then
        flatpak install --user -y flathub "$DRIVERS_NAME//$DRIVERS_BRANCH" &> /dev/null
    else
        flatpak update --user -y "$DRIVERS_NAME//$DRIVERS_BRANCH" &> /dev/null
    fi
    flatpak build-bundle --runtime --runtime-repo=${FLATHUB_REPO} ~/.local/share/flatpak/repo "${INSTALL_PATH}/drivers.flatpak" "$DRIVERS_NAME" "$DRIVERS_BRANCH"
    if [ $runtime_not_installed -eq 1 ]
    then
        flatpak uninstall --user -y "$DRIVERS_NAME//$DRIVERS_BRANCH" &> /dev/null
    fi

    # Export INENDI Inspector bundle
    echo "[3/3] Exporting INENDI Inspector bundle ..."
    flatpak info "${INSPECTOR_NAME}"  &> /dev/null
    inspector_not_installed=$?
    if [ $inspector_not_installed  -eq 1 ]
    then
        flatpak install --user -y inendi_tmp "${INSPECTOR_NAME}" &> /dev/null
    else
        flatpak update --user -y "${INSPECTOR_NAME}" &> /dev/null
    fi
    flatpak build-bundle --runtime-repo=${FLATHUB_REPO} ~/.local/share/flatpak/repo "${INSTALL_PATH}/inendi-inspector.flatpak" "${INSPECTOR_NAME}"
    if [ $inspector_not_installed -eq 1 ]
    then
        flatpak uninstall --user -y "${INSPECTOR_NAME}" &> /dev/null
    fi

    flatpak remote-delete --user inendi_tmp

    wget $NICE_DCV_URL -o "${INSTALL_PATH}"
    aws --no-sign-request s3 sync s3://inendi-inspector "${INSTALL_PATH}"
fi

if [ "${install_mode}" == "online" ]
then
    wget $NICE_DCV_URL -o "${INSTALL_PATH}"
    aws --no-sign-request s3 sync s3://inendi-inspector "${INSTALL_PATH}"
fi
