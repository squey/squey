#!/bin/bash

: "${DOCKER:=docker}"

command -v "${DOCKER}" &> /dev/null || { echo >&2 "'${DOCKER}' executable is required to execute this script."; exit 1; }

${DOCKER} ${DOCKER_OPTS} run --privileged --runtime=nvidia --rm --name inendi-inspector -v /sys/fs/cgroup:/sys/fs/cgroup:ro -p 443:443 -d inendi/inspector
