#!/bin/bash

if [ $# -ne 1 ]; then
	echo "usage : `basename $0` <flatpakref URL>"
    exit 1
fi

apt-get update
apt-get -y remove --purge git* lxd* *openssh* man* krb5* nano mlocate parted dosfstools rsyslog busybox* iptables
apt-get -y autoremove
apt-get install -y ca-certificates flatpak dbus-x11
apt-get -y dist-upgrade
apt-get clean
useradd -m inendi
mkdir -p /srv/tmp-inspector
chown inendi: -R /srv/tmp-inspector
service dbus start
su - inendi -c "flatpak install --user -y $1"
