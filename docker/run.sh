#!/bin/bash

source env.conf
source resources/.env.conf

: "${DOCKER:=docker}"

command -v "${DOCKER}" &> /dev/null || { echo >&2 "'${DOCKER}' executable is required to execute this script."; exit 1; }

NVIDIA_DOCKER_RUNTIME=""
if [ ! -z ${GL_DRIVERS_VERSION} ]
then
    source resources/check_nvidia-docker.sh
fi

${DOCKER} ${DOCKER_OPTS} run --privileged ${NVIDIA_DOCKER_RUNTIME} --name inendi-inspector -v /sys/fs/cgroup:/sys/fs/cgroup:ro -p 443:443 -d --restart unless-stopped inendi/inspector
