#!/bin/bash

source env.conf
source resources/.env.conf

: "${DOCKER:=podman}"

command -v "${DOCKER}" &> /dev/null || { echo >&2 "'${DOCKER}' executable is required to execute this script."; exit 1; }

${DOCKER} volume create squey_flatpak_system_data &> /dev/null || true
${DOCKER} ${DOCKER_OPTS} run \
    --name squey \
    --privileged \
    --tmpfs /run \
    --tmpfs /tmp \
    --mount type=volume,source=squey_flatpak_system_data,target=/var/lib/flatpak \
    -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
    --cgroupns=host \
    -p 8443:443 \
    -d \
    --rm \
    squey/squey
