#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${DIR}/.env.conf"

install_mode="$1"
if [[ ${install_mode} == "offline" ]]
then
    flatpak uninstall -y "${RUNTIME_NAME}"  &> /dev/null
    flatpak install -y "${DIR}/runtime.flatpak"
    flatpak uninstall -y "${INSPECTOR_NAME}" &> /dev/null
    flatpak install -y "${DIR}/inendi-inspector.flatpak"
    rm -rf "${DIR}"/*.flatpak
else # online installation
    flatpak install -y --no-related https://dl.flathub.org/repo/appstream/com.gitlab.inendi.Inspector.flatpakref
    flatpak update -y
fi
