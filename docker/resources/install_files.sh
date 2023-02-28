#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${DIR}/.env.conf"

install_mode="$1"
if [[ ${install_mode} == "offline" ]]
then
    flatpak uninstall -y "${RUNTIME_NAME}"  &> /dev/null
    flatpak install -y "${DIR}/runtime.flatpak"
    flatpak uninstall -y "${SDK_NAME}"  &> /dev/null
    flatpak install -y "${DIR}/sdk.flatpak"
    if [ -f "${DIR}/drivers.flatpak" ]
    then
        flatpak uninstall -y "${DRIVER_NAME}"  &> /dev/null
        flatpak install -y "${DIR}/drivers.flatpak"
    fi
    flatpak uninstall -y "${INSPECTOR_NAME}" &> /dev/null
    flatpak install -y "${DIR}/inendi-inspector.flatpak"
    rm -rf "${DIR}"/*.flatpak
else # online installation
    flatpak install -y https://dl.flathub.org/repo/appstream/com.gitlab.inendi.Inspector.flatpakref
    flatpak install -y flathub "$SDK_NAME//$RUNTIME_BRANCH"
    if [ ! -z ${GL_DRIVERS_VERSION} ]
    then
        flatpak install -y flathub "$DRIVERS_NAME//$DRIVERS_BRANCH"
    fi
    flatpak update -y
fi
