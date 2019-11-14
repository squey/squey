#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

apt update && apt -y install software-properties-common && add-apt-repository ppa:alexlarsson/flatpak && apt install -y flatpak && rm -rf /var/lib/apt/lists/*

install_mode="$1"
if [[ ${install_mode} == "offline" ]]
then
    flatpak uninstall -y "${INSPECTOR_NAME}" &> /dev/null
    flatpak install -y "${DIR}/inendi-inspector.flatpak"
    if [ -f "${DIR}/drivers.flatpak" ]
    then
        flatpak uninstall -y "${DRIVER_NAME}"  &> /dev/null
        flatpak install -y "${DIR}/drivers.flatpak"
    fi
    flatpak uninstall -y "${RUNTIME_NAME}"  &> /dev/null
    flatpak install -y "${DIR}/runtime.flatpak"
    rm -rf "${DIR}/*.flatpak"
else # online installation
    flatpak install -y https://repo.esi-inendi.com/inendi-inspector.flatpakref
    flatpak update -y
fi
