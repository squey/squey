#!/bin/bash

source env.conf
source .env.conf

INSTALL_MODE="online"
[[ -d "data" ]] && INSTALL_MODE="offline"

docker run --privileged --runtime=nvidia --rm --name inspector-update -v /sys/fs/cgroup:/sys/fs/cgroup:ro -d inendi/inspector

if [[ ${INSTALL_MODE} == "offline" ]]
then
    docker cp data/runtime.flatpak inspector-update:"${INSTALL_PATH}/"
    docker cp data/drivers.flatpak inspector-update:"${INSTALL_PATH}/"
    docker cp data/inendi-inspector.flatpak inspector-update:"${INSTALL_PATH}/"
fi

docker exec inspector-update bash "${INSTALL_PATH}/install_files.sh" "${INSTALL_MODE}"
docker commit inspector-update inendi/inspector
docker stop inspector-update
