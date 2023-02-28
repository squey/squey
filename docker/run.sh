#!/bin/bash

source env.conf
source resources/.env.conf

: "${DOCKER:=podman}"

command -v "${DOCKER}" &> /dev/null || { echo >&2 "'${DOCKER}' executable is required to execute this script."; exit 1; }

NVIDIA_DOCKER_RUNTIME=""
if [ ! -z ${GL_DRIVERS_VERSION} ]
then
    source resources/check_nvidia-docker.sh
fi

#${DOCKER} ${DOCKER_OPTS} run --privileged ${NVIDIA_DOCKER_RUNTIME} --name inendi-inspector --tmpfs /run --tmpfs /tmp -v /sys/fs/cgroup:/sys/fs/cgroup:rw --cgroupns=host -p 443:443 -d --restart unless-stopped inendi/inspector
${DOCKER} ${DOCKER_OPTS} run --privileged ${NVIDIA_DOCKER_RUNTIME} --name inendi-inspector --rm --tmpfs /run --tmpfs /tmp -v /sys/fs/cgroup:/sys/fs/cgroup:rw --cgroupns=host -p 8443:443 inendi/inspector

#TODO : use flatpak dir as a volume