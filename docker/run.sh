#!/bin/bash

: "${DOCKER:=docker}"

command -v "${DOCKER}" &> /dev/null || { echo >&2 "'${DOCKER}' executable is required to execute this script."; exit 1; }

${DOCKER} ${DOCKER_OPTS} run --privileged --runtime=nvidia --rm --name inendi-inspector -v /sys/fs/cgroup:/sys/fs/cgroup:ro -p 443:443 -d inendi/inspector
# systemd-docker run --privileged --runtime=nvidia --rm --name inendi-inspector -v /sys/fs/cgroup:/sys/fs/cgroup:ro -p 443:443 -d inendi/inspector # WORKS !

# dbus-uuidgen > /var/lib/dbus/machine-id
# mkdir -p /var/run/dbus
# rm -rf /var/run/dbus/pid
# dbus-daemon --config-file=/usr/share/dbus-1/system.conf --print-address

#/usr/bin/install -d -m 3777 -o dcv -g dcv /var/run/dcvsimpleextauth
# apt-get -y install systemd-docker
