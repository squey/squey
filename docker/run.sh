#!/bin/bash

#docker run -it --privileged -p 8443:8443 inendi/inspector
# dbus-daemon --config-file=/usr/share/dbus-1/system.conf --print-address
# /usr/bin/dcv create-session --type=virtual --user=inspector --owner=inspector --gl on --init /etc/dcv/dcvsessioninit dcv_session

# docker run --privileged -it --rm --name systemd-inspector --cap-add SYS_ADMIN --security-opt seccomp=unconfined --security-opt apparmor=unconfined  -v /sys/fs/cgroup:/sys/fs/cgroup:ro -v /run:/run inendi/inspector /lib/systemd/systemd
# docker exec -it systemd-inspector bash

# GOOD
# echo "kernel.unprivileged_userns_clone = 1" | sudo tee -a /etc/sysctl.d/00-local-userns.conf && sudo sysctl --system
#docker run --cap-add SYS_ADMIN --cap-add SYS_CHROOT --security-opt apparmor=unconfined --security-opt seccomp=unconfined --runtime=nvidia --rm --name inendi-inspector -v /sys/fs/cgroup:/sys/fs/cgroup:ro -p 8443:8443 -d inendi/inspector
docker run --privileged --runtime=nvidia --rm --name inendi-inspector -v /sys/fs/cgroup:/sys/fs/cgroup:ro -p 8443:8443 -d inendi/inspector
